from flask import Flask, Response
import cv2
import time

class CameraStream:
    def __init__(self, camera_url):
        self.camera_url = camera_url
        self.app = Flask(__name__)
        self.setup_routes()
        
    def generate_frames(self):
        while True:  # Outer loop for reconnection
            try:
                cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    print("Failed to open camera stream. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                    
                while True:  # Inner loop for frame reading
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame. Reconnecting...")
                        break
                        
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        print("Failed to encode frame. Skipping...")
                        continue
                        
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
            except Exception as e:
                print(f"Stream error: {str(e)}. Reconnecting...")
                time.sleep(5)
                
            finally:
                cap.release()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return """
            <html>
            <body>
                <h1>People Counter Camera Feed</h1>
                <img src="/video_feed" width="640" height="480" />
            </body>
            </html>
            """

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port) 