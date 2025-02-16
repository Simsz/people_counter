from src.stream import CameraStream
from src.detector import PersonDetector
from src.line_crossing import LineCounter
from dotenv import load_dotenv
import os
import cv2

def main():
    # Load environment variables
    load_dotenv()

    # Configuration from environment variables
    CAMERA_URL = os.getenv('CAMERA_URL')
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/person_detection_edgetpu.tflite')

    if not CAMERA_URL:
        raise ValueError("CAMERA_URL not found in environment variables")

    # Initialize video capture to get frame dimensions
    cap = cv2.VideoCapture(CAMERA_URL)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read from camera")
    
    height, width = frame.shape[:2]
    cap.release()

    # Define counting line (adjust these coordinates based on your camera view)
    # Vertical line through the door
    line_start = (int(width * 0.26), int(height * 0.65))  # Top of door
    line_end = (int(width * 0.26), int(height * 0.88))    # Bottom of door

    # Initialize detector and counter
    detector = PersonDetector(MODEL_PATH)
    counter = LineCounter(line_start, line_end, line_offset=60)  # Wider offset for side detection
    
    # Initialize and run the camera stream
    stream = CameraStream(CAMERA_URL, detector, counter)
    stream.run(host=HOST, port=PORT)

if __name__ == "__main__":
    main() 