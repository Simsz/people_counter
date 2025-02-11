from src.stream import CameraStream
from src.detector import PersonDetector
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()

    # Configuration from environment variables
    CAMERA_URL = os.getenv('CAMERA_URL')
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/detect_edgetpu.tflite')

    if not CAMERA_URL:
        raise ValueError("CAMERA_URL not found in environment variables")

    # Initialize detector
    detector = PersonDetector(MODEL_PATH)
    
    # Initialize and run the camera stream
    stream = CameraStream(CAMERA_URL, detector)
    stream.run(host=HOST, port=PORT)

if __name__ == "__main__":
    main() 