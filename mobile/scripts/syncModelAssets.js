const fs = require("fs");
const path = require("path");

const mobileRoot = path.resolve(__dirname, "..");
const repositoryRoot = path.resolve(mobileRoot, "..");
const sourceModelPath = path.join(
  repositoryRoot,
  "models",
  "v1",
  "mobile-detector.tflite",
);
const generatedModelDirectory = path.join(
  mobileRoot,
  "generated-assets",
  "models",
);
const generatedModelPath = path.join(
  generatedModelDirectory,
  "mobile-detector.tflite",
);

if (!fs.existsSync(sourceModelPath)) {
  throw new Error(`Mobile detector not found: ${sourceModelPath}`);
}

fs.mkdirSync(generatedModelDirectory, { recursive: true });
fs.copyFileSync(sourceModelPath, generatedModelPath);

console.log(`Synced mobile detector: ${generatedModelPath}`);
