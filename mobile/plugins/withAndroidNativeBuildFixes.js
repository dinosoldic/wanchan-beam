const {
  withAppBuildGradle,
  withGradleProperties,
  withProjectBuildGradle,
} = require("expo/config-plugins");

const GENERATED_START = "// @generated wanchan-native-build-fixes start";
const GENERATED_END = "// @generated wanchan-native-build-fixes end";
const RELEASE_SIGNING_START =
  "// @generated wanchan-release-signing start";
const RELEASE_SIGNING_END = "// @generated wanchan-release-signing end";

const RELEASE_SIGNING_BLOCK = `        ${RELEASE_SIGNING_START}
        release {
            storeFile file(findProperty("WANCHAN_UPLOAD_STORE_FILE") ?: "missing-release.keystore")
            storePassword findProperty("WANCHAN_UPLOAD_STORE_PASSWORD") ?: ""
            keyAlias findProperty("WANCHAN_UPLOAD_KEY_ALIAS") ?: ""
            keyPassword findProperty("WANCHAN_UPLOAD_KEY_PASSWORD") ?: ""
        }
        ${RELEASE_SIGNING_END}
`;

const PROJECT_BUILD_BLOCK = `${GENERATED_START}
def wanchanNativeLibraryOwners = [
  'libNitroModules.so': 'react-native-nitro-modules',
  'libNitroImage.so'  : 'react-native-nitro-image',
  'libVisionCamera.so': 'react-native-vision-camera',
  'libworklets.so'    : 'react-native-worklets',
]
def wanchanIsWindows = System.getProperty('os.name').toLowerCase().contains('windows')

// Keep the two longest native staging paths below Windows' legacy limit.
subprojects { subproject ->
  if (wanchanIsWindows && subproject.name in [
    'react-native-vision-camera-resizer',
    'react-native-vision-camera-worklets',
  ]) {
    subproject.plugins.withId('com.android.library') {
      subproject.android.externalNativeBuild.cmake.buildStagingDirectory =
          new File(rootProject.projectDir, ".cxx/\${subproject.name}")
    }
  }

  subproject.afterEvaluate {
    if (subproject.plugins.hasPlugin('com.android.library')) {
      wanchanNativeLibraryOwners.each { libraryName, ownerProject ->
        if (subproject.name != ownerProject) {
          // Wrappers link shared runtimes but only the owning module packages them.
          subproject.android.packagingOptions.exclude "**/\${libraryName}"
        }
      }
    }
  }
}
${GENERATED_END}`;

function replaceActiveGradleProperty(contents, propertyName, value) {
  const propertyPattern = new RegExp(`^(\\s*)${propertyName} = .*?$`, "m");

  if (!propertyPattern.test(contents)) {
    throw new Error(`Could not find ${propertyName} in android/app/build.gradle.`);
  }

  return contents.replace(propertyPattern, (_, indentation) => {
    return `${indentation}${propertyName} = ${value}`;
  });
}

function replaceOrInsertActiveGradleProperty(
  contents,
  propertyName,
  value,
  insertAfterPropertyName,
) {
  const propertyPattern = new RegExp(`^(\\s*)${propertyName} = .*?$`, "m");
  if (propertyPattern.test(contents)) {
    return replaceActiveGradleProperty(contents, propertyName, value);
  }

  const anchorPattern = new RegExp(
    `^(\\s*)${insertAfterPropertyName} = .*?$`,
    "m",
  );
  if (!anchorPattern.test(contents)) {
    throw new Error(
      `Could not find ${propertyName} or its ${insertAfterPropertyName} insertion point in android/app/build.gradle.`,
    );
  }

  return contents.replace(anchorPattern, (anchorLine, indentation) => {
    return `${anchorLine}\n${indentation}${propertyName} = ${value}`;
  });
}

function configureReleaseSigning(contents) {
  const existingStart = contents.indexOf(RELEASE_SIGNING_START);
  if (existingStart >= 0) {
    const existingEnd = contents.indexOf(RELEASE_SIGNING_END, existingStart);
    if (existingEnd < 0) {
      throw new Error("Found an incomplete Android release signing block.");
    }

    const afterExistingEnd = existingEnd + RELEASE_SIGNING_END.length;
    contents = `${contents.slice(0, existingStart)}${contents.slice(
      afterExistingEnd,
    )}`;
  }

  const signingConfigsStart = contents.indexOf("    signingConfigs {");
  const buildTypesStart = contents.indexOf("    buildTypes {");
  if (signingConfigsStart < 0 || buildTypesStart < signingConfigsStart) {
    throw new Error("Could not find Android signingConfigs and buildTypes.");
  }

  const signingConfigsEnd = contents.lastIndexOf("\n    }", buildTypesStart);
  if (signingConfigsEnd < signingConfigsStart) {
    throw new Error("Could not find the end of Android signingConfigs.");
  }

  contents = `${contents.slice(0, signingConfigsEnd)}\n${RELEASE_SIGNING_BLOCK}${contents.slice(
    signingConfigsEnd,
  )}`;

  const updatedBuildTypesStart = contents.indexOf("    buildTypes {");
  const releaseStart = contents.indexOf(
    "        release {",
    updatedBuildTypesStart,
  );
  const releaseEnd = contents.indexOf("\n        }", releaseStart);
  if (releaseStart < 0 || releaseEnd < releaseStart) {
    throw new Error("Could not find the Android release build type.");
  }

  const releaseBlock = contents.slice(releaseStart, releaseEnd);
  const debugSigning = "signingConfig signingConfigs.debug";
  const releaseSigning = "signingConfig signingConfigs.release";
  if (
    !releaseBlock.includes(debugSigning) &&
    !releaseBlock.includes(releaseSigning)
  ) {
    throw new Error("Could not find the Android release signing selection.");
  }

  const updatedReleaseBlock = releaseBlock.replace(debugSigning, releaseSigning);
  return `${contents.slice(0, releaseStart)}${updatedReleaseBlock}${contents.slice(
    releaseEnd,
  )}`;
}

function removeGeneratedProjectBlock(contents) {
  const startIndex = contents.indexOf(GENERATED_START);
  if (startIndex === -1) {
    return contents;
  }

  const endIndex = contents.indexOf(GENERATED_END, startIndex);
  if (endIndex === -1) {
    throw new Error("Found an incomplete WanChan native build block.");
  }

  return `${contents.slice(0, startIndex).trimEnd()}\n\n${contents
    .slice(endIndex + GENERATED_END.length)
    .trimStart()}`;
}

function removeLegacyProjectBlock(contents) {
  const startIndex = contents.indexOf("def nativeLibraryOwners = [");
  if (startIndex === -1) {
    return contents;
  }

  const applyPluginIndex = contents.indexOf(
    'apply plugin: "expo-root-project"',
    startIndex,
  );
  if (applyPluginIndex === -1) {
    throw new Error("Could not locate the end of the legacy native build block.");
  }

  return `${contents.slice(0, startIndex).trimEnd()}\n\n${contents.slice(
    applyPluginIndex,
  )}`;
}

function withPersistentAndroidNativeBuildFixes(config) {
  config = withProjectBuildGradle(config, (projectConfig) => {
    if (projectConfig.modResults.language !== "groovy") {
      throw new Error("WanChan Android build fixes require Groovy Gradle files.");
    }

    let contents = removeGeneratedProjectBlock(projectConfig.modResults.contents);
    contents = removeLegacyProjectBlock(contents);

    const insertionPoint = 'apply plugin: "expo-root-project"';
    if (!contents.includes(insertionPoint)) {
      throw new Error("Could not find the Android root plugin insertion point.");
    }

    projectConfig.modResults.contents = contents.replace(
      insertionPoint,
      `${PROJECT_BUILD_BLOCK}\n\n${insertionPoint}`,
    );
    return projectConfig;
  });

  config = withAppBuildGradle(config, (appConfig) => {
    if (appConfig.modResults.language !== "groovy") {
      throw new Error("WanChan Android app fixes require Groovy Gradle files.");
    }

    let contents = appConfig.modResults.contents;
    contents = replaceActiveGradleProperty(
      contents,
      "reactNativeDir",
      `new File(["node", "--print", "require('fs').realpathSync.native(require.resolve('react-native/package.json'))"].execute(null, rootDir).text.trim()).getParentFile().getAbsoluteFile()`,
    );
    contents = replaceActiveGradleProperty(
      contents,
      "codegenDir",
      `new File(["node", "--print", "require('fs').realpathSync.native(require.resolve('@react-native/codegen/package.json', { paths: [require.resolve('react-native/package.json')] }))"].execute(null, rootDir).text.trim()).getParentFile().getAbsoluteFile()`,
    );
    contents = replaceActiveGradleProperty(
      contents,
      "hermesCommand",
      `new File(["node", "--print", "require('fs').realpathSync.native(require.resolve('hermes-compiler/package.json', { paths: [require.resolve('react-native/package.json')] }))"].execute(null, rootDir).text.trim()).getParentFile().getAbsolutePath() + "/hermesc/%OS-BIN%/hermesc"`,
    );
    contents = replaceOrInsertActiveGradleProperty(
      contents,
      "hermesFlags",
      '["-O"]',
      "hermesCommand",
    );
    contents = configureReleaseSigning(contents);

    appConfig.modResults.contents = contents;
    return appConfig;
  });

  return withGradleProperties(config, (gradleConfig) => {
    const existingPropertyIndex = gradleConfig.modResults.findIndex(
      (item) => item.type === "property" && item.key === "kotlin.incremental",
    );
    const existingProperty = gradleConfig.modResults[existingPropertyIndex];

    if (process.platform === "win32" && existingProperty) {
      existingProperty.value = "false";
    } else if (process.platform === "win32") {
      gradleConfig.modResults.push({
        type: "property",
        key: "kotlin.incremental",
        value: "false",
      });
    } else if (existingPropertyIndex >= 0) {
      gradleConfig.modResults.splice(existingPropertyIndex, 1);
    }

    return gradleConfig;
  });
}

module.exports = withPersistentAndroidNativeBuildFixes;
