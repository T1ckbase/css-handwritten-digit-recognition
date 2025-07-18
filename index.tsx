import type { FC } from 'hono/jsx';

const width = 28;
const height = 28;

interface ModelWeights {
  'fc.weight': number[][];
  'fc.bias': number[];
}

async function generateCssFromWeights(jsonPath: string, cssPath: string): Promise<void> {
  try {
    const weightsJson = await Deno.readTextFile(jsonPath);
    const weights: ModelWeights = JSON.parse(weightsJson);

    const inputSize = weights['fc.weight']?.[0]?.length;
    const outputSize = weights['fc.weight']?.length;

    if (
      !inputSize || !outputSize ||
      weights['fc.bias']?.length !== outputSize
    ) {
      console.error('Error: Weight dimensions are missing, malformed, or inconsistent.');
      return;
    }

    // Use an array to build the CSS content for better performance.
    const cssContent: string[] = [];

    // 1. Define input variables.
    cssContent.push(`/* 1. INPUT LAYER (${inputSize} variables) */`);
    for (let i = 0; i < inputSize; i++) {
      cssContent.push(`@property --in-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
    }

    // Calculate logits directly from inputs
    cssContent.push(`\n/* 2. OUTPUT LAYER (${outputSize} neurons) - Logits */`);
    for (let i = 0; i < outputSize; i++) {
      const logitTerms = weights['fc.weight'][i].map((weight, j) => {
        return `var(--in-${j}) * ${weight}`;
      });
      const bias = weights['fc.bias'][i];
      cssContent.push(`@property --logit-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
      cssContent.push(`:root { --logit-${i}: calc(${logitTerms.join(' + ')} + ${bias}); }`);
      // cssContent.push(`@property --out-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
      // cssContent.push(`:root { --out-${i}: calc(${logitTerms.join(' + ')} + ${bias}); }`);
    }

    // 4. Final Softmax Activation Layer.
    cssContent.push(`\n/* 4. SOFTMAX ACTIVATION */`);

    cssContent.push(`/* Exponentiate logits */`);
    for (let i = 0; i < outputSize; i++) {
      cssContent.push(`@property --exp-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
      cssContent.push(`:root { --exp-${i}: exp(var(--logit-${i})); }`);
    }

    cssContent.push(`\n/* Sum of all exponentiated logits */`);
    const expVars = Array.from({ length: outputSize }, (_, i) => `var(--exp-${i})`);
    cssContent.push(`@property --exp-sum { syntax: "<number>"; inherits: true; initial-value: 0; }`);
    cssContent.push(`:root { --exp-sum: calc(${expVars.join(' + ')}); }`);

    cssContent.push(`\n/* Final Probabilities (output) */`);
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

await generateCssFromWeights('./mnist_model_weights.json', 'model.css');
// Deno.exit();

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
      <title>Pure CSS Handwritten Digit Recognition</title>
      <link rel='stylesheet' href='./model.css' />
      <link rel='stylesheet' href='./main.css' />
    </head>
    <body>
      <Grid width={width} height={height} />
      <div class='debug0'>debug:</div>
      <div class='debug1'></div>
      {/* <div class='test'></div> */}
    </body>
  </html>
);
const html = `<!DOCTYPE html>${jsxElement.toString()}`;
await Deno.writeTextFile('./index.html', html);
