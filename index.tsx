import type { FC } from 'hono/jsx';
import * as sass from 'sass';

const width = 28;
const height = 28;

interface ModelWeights {
  'fc1.weight': number[][];
  'fc1.bias': number[];
  'fc2.weight': number[][];
  'fc2.bias': number[];
}

async function generateCssFromWeights(jsonPath: string, cssPath: string): Promise<void> {
  try {
    const weightsJson = await Deno.readTextFile(jsonPath);
    const weights: ModelWeights = JSON.parse(weightsJson);

    let inputSize = weights['fc1.weight']?.[0]?.length;
    const hiddenSize = weights['fc1.weight']?.length;
    const outputSize = weights['fc2.weight']?.length;

    if (
      !inputSize || !hiddenSize || !outputSize ||
      weights['fc1.bias']?.length !== hiddenSize ||
      weights['fc2.weight']?.[0]?.length !== hiddenSize || // Crucial check: hidden layer output must match next layer's input
      weights['fc2.bias']?.length !== outputSize
    ) {
      console.error('Error: Weight dimensions are missing, malformed, or inconsistent for a 3-layer network.');
      return;
    }
    // inputSize = 10;

    // Use an array to build the CSS content for better performance.
    const cssContent: string[] = [];

    // 1. Define input variables.
    cssContent.push(`/* 1. INPUT LAYER (${inputSize} variables) */`);
    for (let i = 0; i < inputSize; i++) {
      cssContent.push(`@property --in-${i} { syntax: "<number>"; inherits: true; initial-value: 0; }`);
    }

    // 2. Calculate hidden layer.
    cssContent.push(`\n/* 2. HIDDEN LAYER (${hiddenSize} neurons) */`);
    for (let i = 0; i < hiddenSize; i++) {
      const preActivationTerms = weights['fc1.weight'][i].map((weight, j) => {
        return `var(--in-${j}) * ${weight}`;
      });
      const bias = weights['fc1.bias'][i];
      cssContent.push(`:root { --h-pre-${i}: calc(${preActivationTerms.join(' + ')} + ${bias}); }`);
    }

    // Apply ReLU activation.
    cssContent.push('\n/* ReLU Activation */');
    for (let i = 0; i < hiddenSize; i++) {
      cssContent.push(`:root { --h-out-${i}: max(0, var(--h-pre-${i})); }`);
    }

    // 3. Calculate output layer (logits).
    cssContent.push(`\n/* 3. OUTPUT LAYER (${outputSize} neurons) - Logits */`);
    for (let i = 0; i < outputSize; i++) {
      const outputTerms = weights['fc2.weight'][i].map((weight, j) => {
        return `var(--h-out-${j}) * ${weight}`;
      });
      const bias = weights['fc2.bias'][i];
      cssContent.push(`:root { --logit-${i}: calc(${outputTerms.join(' + ')} + ${bias}); }`);
    }

    // 4. Final Softmax Activation Layer.
    cssContent.push(`\n/* 4. SOFTMAX ACTIVATION */`);

    cssContent.push(`/* Exponentiate logits */`);
    for (let i = 0; i < outputSize; i++) {
      cssContent.push(`:root { --exp-${i}: exp(var(--logit-${i})); }`);
    }

    cssContent.push(`\n/* Sum of all exponentiated logits */`);
    const expVars = Array.from({ length: outputSize }, (_, i) => `var(--exp-${i})`);
    cssContent.push(`:root { --exp-sum: calc(${expVars.join(' + ')}); }`);

    cssContent.push(`\n/* Final Probabilities (output) */`);
    for (let i = 0; i < outputSize; i++) {
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
Deno.exit();

const Layout: FC = (props) => {
  return (
    <html lang='en'>
      <head>
        <meta charset='UTF-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1.0' />
        <meta name='color-scheme' content='dark' />
        <title>Pure CSS Handwritten Digit Recognition</title>
        <link rel='stylesheet' href='./model.css' />
        <link rel='stylesheet' href='./main.css' />
      </head>
      <body>{props.children}</body>
    </html>
  );
};

const Grid: FC<{ width: number; height: number }> = ({ width, height }) => {
  const cells = Array.from({ length: width * height }, (_, i) => <div class={`cell cell-${i}`}></div>);
  return <div class='grid' style={{ width: `calc(var(--cell-size) * ${width})` }}>{cells}</div>;
};

const App: FC = (props) => {
  return (
    <>
      {/* {Array.from({ length: 10000 }, (_, i) => <div>hello world</div>)} */}
      <Grid width={width} height={height} />
      {/* <div class='mandelbrot-set'></div> */}
      {/* <NestedDiv className='line' count={maxIterations} /> */}
      <div class='debug'>debug:</div>
    </>
  );
};

const jsxElement = (
  <Layout>
    <App />
  </Layout>
);
const html = `<!DOCTYPE html>${jsxElement.toString()}`;
await Deno.writeTextFile('./index.html', html);

const scssContent = await Deno.readTextFile('main.scss');
const { css } = sass.compileString(scssContent /* , { style: 'compressed' } */);
await Deno.writeTextFile('./main.css', css);
