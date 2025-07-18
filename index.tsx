import type { FC } from 'hono/jsx';

const width = 28;
const height = 28;

interface ModelWeights {
  'conv_layer.0.weight': number[][][][]; // [out_channels, in_channels, kernel_h, kernel_w]
  'conv_layer.0.bias': number[];
  'fc_layer.weight': number[][];
  'fc_layer.bias': number[];
}

async function generateCnnCss(jsonPath: string, cssPath: string): Promise<void> {
  try {
    const weightsJson = await Deno.readTextFile(jsonPath);
    const weights: ModelWeights = JSON.parse(weightsJson);

    // --- Automatically detect architecture parameters ---
    const convWeights = weights['conv_layer.0.weight'];
    const fcWeights = weights['fc_layer.weight'];

    const outChannels = convWeights.length;
    const inChannels = convWeights[0].length;
    const kernelH = convWeights[0][0].length;
    const kernelW = convWeights[0][0][0].length;
    const convBiasCount = weights['conv_layer.0.bias'].length;

    const outputSize = fcWeights.length;
    const fcInputSize = fcWeights[0].length;
    const fcBiasCount = weights['fc_layer.bias'].length;

    // --- Hardcoded parameters for MNIST and the specific CNN architecture ---
    const imgH = 28;
    const imgW = 28;
    const convPadding = 2; // As defined in the Python script for 'same' padding
    const poolKernel = 2; // 2
    const poolStride = 2; // 2

    // --- Validate Architecture ---
    if (inChannels !== 1 || outChannels !== convBiasCount || outputSize !== fcBiasCount) {
      console.error('Error: Mismatch in weight and bias dimensions.');
      return;
    }
    console.log(`Detected Conv Layer: ${inChannels}x${imgH}x${imgW} -> ${outChannels} channels with ${kernelH}x${kernelW} kernel.`);
    console.log(`Detected FC Layer: ${fcInputSize} -> ${outputSize} classes.`);

    // --- Calculate dimensions after each step ---
    const convOutH = Math.floor((imgH - kernelH + 2 * convPadding) / 1) + 1; // Stride=1
    const convOutW = Math.floor((imgW - kernelW + 2 * convPadding) / 1) + 1;

    const poolOutH = Math.floor((convOutH - poolKernel) / poolStride) + 1;
    const poolOutW = Math.floor((convOutW - poolKernel) / poolStride) + 1;

    // Final validation
    if (fcInputSize !== outChannels * poolOutH * poolOutW) {
      console.error(`Error: Flattened dimension mismatch. Expected ${outChannels * poolOutH * poolOutW}, but FC layer input is ${fcInputSize}.`);
      return;
    }

    const cssContent: string[] = [
      `/* Architecture: Conv(${outChannels}x${kernelH}x${kernelW}) -> ReLU -> MaxPool(${poolKernel}x${poolKernel}) -> FC(${fcInputSize}, ${outputSize}) -> Softmax */\n`,
    ];

    // --- 1. INPUT LAYER ---
    cssContent.push(`/* 1. INPUT LAYER (${imgH}x${imgW} variables) */`);
    for (let i = 0; i < imgH * imgW; i++) {
      cssContent.push(`@property --in-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
    }

    // --- 2. CONVOLUTIONAL LAYER ---
    cssContent.push(`\n/* 2. CONVOLUTIONAL LAYER (${outChannels} output channels) */`);
    // For each output channel (filter)
    for (let c_out = 0; c_out < outChannels; c_out++) {
      // For each pixel in the output feature map
      for (let y = 0; y < convOutH; y++) {
        for (let x = 0; x < convOutW; x++) {
          const terms: string[] = [];
          // Apply the kernel
          for (let ky = 0; ky < kernelH; ky++) {
            for (let kx = 0; kx < kernelW; kx++) {
              const inputY = y - convPadding + ky;
              const inputX = x - convPadding + kx;

              // Check if the kernel is over a valid input pixel (not padding)
              if (inputY >= 0 && inputY < imgH && inputX >= 0 && inputX < imgW) {
                const inputIndex = inputY * imgW + inputX;
                const weight = convWeights[c_out][0][ky][kx]; // in_channel is always 0
                terms.push(`var(--in-${inputIndex}) * ${weight}`);
              }
            }
          }
          const bias = weights['conv_layer.0.bias'][c_out];
          const outputPixelIndex = y * convOutW + x;
          cssContent.push(`@property --conv-pre-${c_out}-${outputPixelIndex} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
          cssContent.push(`:root { --conv-pre-${c_out}-${outputPixelIndex}: calc(${terms.join(' + ')} + ${bias}); }`);
        }
      }
    }

    // --- 3. RELU ACTIVATION ---
    cssContent.push(`\n/* 3. RELU ACTIVATION */`);
    for (let c = 0; c < outChannels; c++) {
      for (let i = 0; i < convOutH * convOutW; i++) {
        cssContent.push(`@property --conv-out-${c}-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
        cssContent.push(`:root { --conv-out-${c}-${i}: max(0, var(--conv-pre-${c}-${i})); }`);
      }
    }

    // --- 4. MAX POOLING LAYER ---
    cssContent.push(`\n/* 4. MAX POOLING LAYER */`);
    for (let c = 0; c < outChannels; c++) {
      for (let y = 0; y < poolOutH; y++) {
        for (let x = 0; x < poolOutW; x++) {
          const terms: string[] = [];
          // Apply the pooling window
          for (let py = 0; py < poolKernel; py++) {
            for (let px = 0; px < poolKernel; px++) {
              const inputY = y * poolStride + py;
              const inputX = x * poolStride + px;
              const inputIndex = inputY * convOutW + inputX;
              terms.push(`var(--conv-out-${c}-${inputIndex})`);
            }
          }
          const outputPixelIndex = y * poolOutW + x;
          cssContent.push(`@property --pool-out-${c}-${outputPixelIndex} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
          cssContent.push(`:root { --pool-out-${c}-${outputPixelIndex}: max(${terms.join(', ')}); }`);
        }
      }
    }

    // --- 5. FLATTEN & FULLY CONNECTED LAYER (Logits) ---
    cssContent.push(`\n/* 5. FULLY CONNECTED LAYER (Logits) */`);
    const flattenedVars: string[] = [];
    for (let c = 0; c < outChannels; c++) {
      for (let i = 0; i < poolOutH * poolOutW; i++) {
        flattenedVars.push(`var(--pool-out-${c}-${i})`);
      }
    }

    for (let i = 0; i < outputSize; i++) {
      const fcTerms = fcWeights[i].map((weight, j) => {
        return `${flattenedVars[j]} * ${weight}`;
      });
      const bias = weights['fc_layer.bias'][i];
      cssContent.push(`@property --logit-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
      cssContent.push(`:root { --logit-${i}: calc(${fcTerms.join(' + ')} + ${bias}); }`);
      // cssContent.push(`@property --out-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
      // cssContent.push(`:root { --out-${i}: calc(${fcTerms.join(' + ')} + ${bias}); }`);
    }

    // --- 6. SOFTMAX ACTIVATION ---
    cssContent.push(`\n/* 6. SOFTMAX ACTIVATION */`);
    cssContent.push(`/* 6a. Exponentiate logits */`);
    for (let i = 0; i < outputSize; i++) {
      cssContent.push(`@property --exp-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
      cssContent.push(`:root { --exp-${i}: exp(var(--logit-${i})); }`);
    }

    cssContent.push(`\n/* 6b. Sum of exponentiated logits */`);
    const expVars = Array.from({ length: outputSize }, (_, i) => `var(--exp-${i})`);
    cssContent.push(`@property --exp-sum { syntax: "<number>"; inherits: true; initial-value: 0; }`);
    cssContent.push(`:root { --exp-sum: calc(${expVars.join(' + ')}); }`);

    cssContent.push(`\n/* 6c. Final Probabilities */`);
    for (let i = 0; i < outputSize; i++) {
      cssContent.push(`@property --out-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
      cssContent.push(`:root { --out-${i}: calc(var(--exp-${i}) / var(--exp-sum)); }`);
    }

    await Deno.writeTextFile(cssPath, cssContent.join('\n'));
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) {
      console.error(`Error: The file was not found at ${jsonPath}`);
    } else {
      console.error('An unexpected error occurred:', error instanceof Error ? error.message : error);
    }
  }
}

await generateCnnCss('./mnist_model_weights.json', 'model.css');

const Grid: FC<{ width: number; height: number }> = ({ width, height }) => {
  const cells = Array.from({ length: width * height }, (_, i) => <div class={`cell cell-${i}`}></div>);
  return <div class='grid' style={{ width: `calc(var(--cell-size) * ${width})` }}>{cells}</div>;
};

const jsxElement = (
  <html lang='en'>
    <head>
      <meta charset='UTF-8' />
      <meta name='viewport' content='width=device-width, initial-scale=1.0' />
      <meta name='color-scheme' content='dark' />
      <meta name='description' content='Pure CSS implementation of a CNN for MNIST digit recognition' />
      <meta name='keywords' content='CSS, machine learning, handwritten digit recognition, neural network, front-end AI, web development, MNIST' />
      <title>Pure CSS Handwritten Digit Recognition - MNIST</title>
      <link rel='stylesheet' href='./model.css' />
      <link rel='stylesheet' href='./main.css' />
    </head>
    <body>
      <header>
        <h1>Pure CSS Handwritten Digit Recognition</h1>
        <a href='https://github.com/T1ckbase/css-handwritten-digit-recognition'>GITHUB</a>
      </header>
      <div>Draw a digit in the box.</div>
      <Grid width={width} height={height} />
      <button type='button' class='clear'>clear</button>
      <div class='debug'>debug:</div>
      {/* <div class='debug1'></div> */}
      {/* <div class='test'></div> */}
    </body>
  </html>
);
const html = `<!DOCTYPE html>${jsxElement.toString()}`;
await Deno.writeTextFile('./index.html', html);
