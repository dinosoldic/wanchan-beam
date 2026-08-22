const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// TensorFlow Lite models are bundled as opaque native assets.
config.resolver.assetExts.push("tflite");

module.exports = config;
