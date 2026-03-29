import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize model path
model_path = "face_detector.task"

# Create a face detector instance
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5
)

with FaceDetector.create_from_options(options) as detector:
    # Load an image using OpenCV
    image_path = 'hhh.jpg'  # Change this to your image file
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Run face detection
    detection_result = detector.detect(mp_image)

    # Draw bounding boxes
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image_bgr, start_point, end_point, (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Face Detection', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
