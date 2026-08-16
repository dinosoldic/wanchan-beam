# WanChan Beam

WanChan Beam is a mobile dog-breed scanner and an end-to-end machine-learning
portfolio project. It combines an Expo/React Native client, a Fastify inference
server, PyTorch training code, shared prediction contracts, and versioned ONNX
model packages.

Predictions are probabilistic. Low-confidence results should be presented as
uncertain or unknown, and mixed breeds should not be described as definite
pure-breed identifications.

## Repository structure

- `mobile/` — Expo and React Native application
- `server/` — Fastify and ONNX Runtime inference API
- `ml/` — datasets, training, evaluation, and ONNX export
- `shared/` — cross-system schemas and model metadata contracts
- `models/` — versioned deployable model packages

Development is proceeding incrementally, beginning with the mobile camera
foundation.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
