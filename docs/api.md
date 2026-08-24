# Inference API

The hosted API exposes a health check and one image inference endpoint. Its JSON
response contract is defined by the schemas in `server/schemas` and used by
Fastify when serializing results.

Default local base URL:

```text
http://localhost:3000
```

## Health check

```http
GET /health
```

Successful response:

```json
{
  "status": "ok"
}
```

The health check confirms that the process started and both ONNX models passed
startup loading.

## Detect dogs

```http
POST /detect
Content-Type: multipart/form-data
```

Send exactly one file in a multipart field named `image`.

Accepted formats:

- JPEG
- PNG
- WebP

Maximum file size: 10 MB.

Example:

```bash
curl -F "image=@dogs.jpg;type=image/jpeg" http://localhost:3000/detect
```

Successful response:

```json
{
  "image": {
    "width": 1280,
    "height": 720
  },
  "detections": [
    {
      "classId": 16,
      "label": "dog",
      "confidence": 0.91,
      "box": {
        "x1": 184.2,
        "y1": 96.5,
        "x2": 612.8,
        "y2": 681.1
      },
      "breedPredictions": [
        {
          "classId": 5,
          "label": "Australian_Shepherd",
          "confidence": 0.82
        },
        {
          "classId": 98,
          "label": "collie",
          "confidence": 0.07
        }
      ]
    }
  ]
}
```

`image.width` and `image.height` are the auto-oriented original dimensions.
Box values are floating-point pixel coordinates in that same image space:

- `(x1, y1)` is the top-left corner;
- `(x2, y2)` is the bottom-right corner; and
- values are clamped to the image boundaries.

`confidence` on the detection is the detector score. Each breed prediction has
its own classifier confidence. `breedPredictions` always contains the ordered
top two classifier results, even when the app later displays the first result
as uncertain.

An image with no detected dogs is still successful:

```json
{
  "image": {
    "width": 1280,
    "height": 720
  },
  "detections": []
}
```

## Errors

Expected client errors share one shape:

```json
{
  "error": "An image file is required"
}
```

| Status | Meaning |
| ---: | --- |
| `400` | Missing file or multipart field is not named `image` |
| `413` | Uploaded file exceeds 10 MB |
| `415` | MIME type is not JPEG, PNG, or WebP |
| `422` | The uploaded bytes could not be decoded as a valid image |

Unexpected inference or infrastructure failures use Fastify's server error
response. The mobile app treats any failed response or its five-second timeout
as a reason to try the bundled on-device models.

## Source of truth

- Route implementation: `server/src/api/detect.ts`
- Response schema: `server/schemas/dog-detection-response.schema.json`
- Error schema: `server/schemas/error-response.schema.json`
- Mobile client: `mobile/services/RemoteInferenceService.ts`
