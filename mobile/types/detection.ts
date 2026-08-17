export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DogDetection {
  classId: 16;
  label: "dog";
  confidence: number;
  box: DetectionBox;
}

export interface DetectionImage {
  width: number;
  height: number;
}

export interface DogDetectionResponse {
  image: DetectionImage;
  detections: DogDetection[];
}
