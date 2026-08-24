# Android releases

Android releases are built and signed by
`.github/workflows/android-release.yml`. The workflow can be run manually for a
test artifact or triggered by a version tag for a public GitHub Release.

The release APK currently targets 64-bit ARM devices (`arm64-v8a`). Building
one architecture keeps the native build smaller and avoids spending runner time
on emulator architectures that are not part of the release.

## Version sources

Keep these values aligned for an app release:

- `mobile/package.json` -> `version`
- `mobile/app.json` -> `expo.version`
- `mobile/app.json` -> `android.versionCode`
- `server/package.json` -> `version`, when the server belongs to the same release

The first two must match or GitHub Actions stops before building. A release tag
must use the same version with a `v` prefix, for example `v1.0.0`.

Increase `android.versionCode` for every Android release, including rebuilds of
an existing marketing version that must install as an update.

If the model package changes, also update its package version and generated
hashes in the relevant manifests.

## GitHub configuration

The workflow expects one repository variable:

- `EXPO_PUBLIC_API_URL`: HTTPS base URL of the hosted inference server

It also expects these repository secrets:

- `ANDROID_KEYSTORE_BASE64`
- `ANDROID_KEYSTORE_PASSWORD`

The configured key alias is `wanchan-beam`, and the current workflow uses the
same secret for the store and key passwords. Secret values and the original
keystore must never be committed.

Keep an encrypted backup of the keystore and its credentials outside GitHub. A
lost signing key prevents future APKs from updating an installed release signed
with that key.

## Pre-release checks

Run these from `mobile`:

```bash
npm ci
npm run sync:model-assets
npm run typecheck
npm run lint
```

Run these from `server` when server code or server model assets changed:

```bash
npm ci
npm run typecheck
npm test
```

Test the main user paths on a physical Android device:

- live camera in portrait and landscape
- several dogs in one frame
- gallery photo scan
- photo captured inside the app
- local fallback with the server unavailable
- uncertain breed label
- save and share actions

## Test a signed workflow build

Use **Actions -> Android release -> Run workflow**. A manual run:

1. installs locked npm dependencies;
2. copies the versioned model package into Metro assets;
3. runs TypeScript and ESLint checks;
4. generates a clean native Android project;
5. builds the signed release APK;
6. verifies its signature; and
7. stores it as a workflow artifact for 14 days.

Manual runs do not create a GitHub Release. Install that artifact on a physical
device before creating the version tag.

## Publish a release

Create the tag only after the signed manual artifact has passed device testing:

```bash
git checkout main
git pull --ff-only
git tag -a v1.0.0 -m "WanChan Beam v1.0.0"
git push origin v1.0.0
```

The tag run verifies that `v1.0.0`, `mobile/package.json`, and
`mobile/app.json` agree. It then creates a GitHub Release with generated notes
and attaches:

```text
wanchan-beam-v1.0.0-android-arm64.apk
```

If the release already exists, the workflow replaces the APK asset for that
tag. Avoid moving a public tag after users may have downloaded it; publish a new
patch version instead.

## Native project note

The `android` directory is generated and ignored. `expo prebuild --clean`
recreates it from `mobile/app.json` and
`mobile/plugins/withAndroidNativeBuildFixes.js`.

The config plugin keeps Windows native staging paths short, prevents duplicate
shared-library packaging, configures release signing, and applies the tested
Node/Hermes paths. Changes made directly inside `mobile/android` will disappear
on the next clean prebuild and should be moved into app config or the plugin.
