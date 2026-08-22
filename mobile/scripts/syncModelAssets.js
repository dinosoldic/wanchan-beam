const fs = require("fs");
const path = require("path");

const mobileRoot = path.resolve(__dirname, "..");
const repositoryRoot = path.resolve(mobileRoot, "..");
const sourceModelDirectory = path.join(repositoryRoot, "models", "v1");
const generatedModelDirectory = path.join(
  mobileRoot,
  "generated-assets",
  "models",
);

const mobileAssetNames = [
  "mobile-detector.tflite",
  "mobile-breed-classifier.tflite",
  "labels.json",
  "preprocessing.json",
];

const mobileAssets = mobileAssetNames.map((assetName) => ({
  assetName,
  sourcePath: path.join(sourceModelDirectory, assetName),
  generatedPath: path.join(generatedModelDirectory, assetName),
}));

// Validate every source before replacing any generated asset.
for (const asset of mobileAssets) {
  if (!fs.existsSync(asset.sourcePath)) {
    throw new Error(`Mobile asset not found: ${asset.sourcePath}`);
  }
}

fs.mkdirSync(generatedModelDirectory, { recursive: true });

for (const asset of mobileAssets) {
  fs.copyFileSync(asset.sourcePath, asset.generatedPath);
  console.log(`Synced mobile asset: ${asset.generatedPath}`);
}
