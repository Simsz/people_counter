from flask import Flask, Response, render_template_string, request
import cv2
import time
import os
from datetime import datetime
from .detector import PersonDetector
import threading
import numpy as np

class CameraStream:
    def __init__(self, camera_url, detector, counter):
        self.camera_url = camera_url
        self.detector = detector
        self.counter = counter
        self.frame = None
        self.processed_frame = None
        self.last_frame_time = 0
        self.fps = 0
        self.running = False
        self.lock = threading.Lock()
        self.app = Flask(__name__)
        self.width = int(os.getenv('STREAM_WIDTH', '1280'))
        self.height = int(os.getenv('STREAM_HEIGHT', '720'))
        
        # Frame rate control
        self.target_fps = 18  # Match camera's FPS
        self.frame_interval = 1.0 / self.target_fps
        self.last_capture_time = 0
        
        # FPS calculation
        self.fps_update_interval = 1.0  # Update FPS every second
        self.frame_count = 0
        self.fps_last_update = time.time()
        
        self.setup_routes()
        
    def capture_frames(self):
        cap = cv2.VideoCapture(self.camera_url)
        self.running = True
        
        while self.running:
            current_time = time.time()
            
            # Control frame rate
            if current_time - self.last_capture_time < self.frame_interval:
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
                continue
                
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                time.sleep(1)
                continue
            
            self.last_capture_time = current_time
            
            # Update FPS calculation
            self.frame_count += 1
            if current_time - self.fps_last_update >= self.fps_update_interval:
                self.fps = self.frame_count / (current_time - self.fps_last_update)
                self.frame_count = 0
                self.fps_last_update = current_time
            
            # Process frame
            try:
                # Run detection and tracking
                detection_data = self.detector.detect(frame)
                
                # Update counter
                self.counter.update(detection_data[1])  # Pass tracks to counter
                
                # Draw visualizations
                processed = frame.copy()
                processed = self.detector.draw_detections(processed, detection_data)
                processed = self.counter.draw(processed)
                
                # Add FPS counter
                cv2.putText(processed, f"FPS: {self.fps:.1f}/{self.target_fps}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                with self.lock:
                    self.frame = frame
                    self.processed_frame = processed
                    
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                import traceback
                traceback.print_exc()
                
        cap.release()
        
    def get_frame(self):
        while True:
            with self.lock:
                if self.processed_frame is None:
                    continue
                    
                # Encode frame
                _, buffer = cv2.imencode('.jpg', self.processed_frame)
                frame_bytes = buffer.tobytes()
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Match streaming rate to capture rate
            time.sleep(self.frame_interval)
            
    def setup_routes(self):
        def get_available_models():
            models_dir = 'models'
            return [f for f in os.listdir(models_dir) if f.endswith('_edgetpu.tflite')]

        @self.app.route('/switch_model/<model_name>')
        def switch_model(model_name):
            model_path = os.path.join('models', model_name)
            if os.path.exists(model_path):
                self.detector = PersonDetector(model_path)
                return {'success': True, 'model': model_name}
            return {'success': False, 'error': 'Model not found'}, 404

        @self.app.route('/update_roi', methods=['POST'])
        def update_roi():
            data = request.get_json()
            x = int(data['x'])
            y = int(data['y'])
            w = int(data['width'])
            h = int(data['height'])
            self.detector.set_roi([x, y, w, h])
            return {'success': True}

        @self.app.route('/')
        def index():
            models = get_available_models()
            current_model = os.path.basename(self.detector.tpu.model_path)
            
            # Insert the model switcher HTML into the existing template
            model_switcher_html = """
                <div class="info-panel">
                    <h2>Model Selection</h2>
                    <div class="model-switcher">
                        <select id="modelSelect" class="model-select">
                            {% for model in models %}
                                <option value="{{ model }}" {% if model == current_model %}selected{% endif %}>
                                    {{ model }}
                                </option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                <script>
                    document.getElementById('modelSelect').addEventListener('change', function() {
                        fetch('/switch_model/' + this.value)
                            .then(response => response.json())
                            .then(data => {
                                if (data.success) {
                                    console.log('Model switched to: ' + data.model);
                                } else {
                                    console.error('Failed to switch model');
                                }
                            });
                    });
                </script>
            """
            
            # Add these styles to the existing CSS
            additional_styles = """
                .model-switcher {
                    margin-top: 15px;
                }
                
                .model-select {
                    width: 100%;
                    padding: 10px;
                    border-radius: 6px;
                    background: var(--bg-color);
                    color: var(--text-color);
                    border: 1px solid var(--border-color);
                    font-size: 1rem;
                    cursor: pointer;
                }
                
                .model-select:hover {
                    border-color: var(--accent-color);
                }
                
                .model-select option {
                    background: var(--card-bg);
                    color: var(--text-color);
                    padding: 10px;
                }
            """
            
            # Get the existing template
            template = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>People Counter</title>
                <style>
                    :root {
                        --bg-color: #1a1a1a;
                        --text-color: #ffffff;
                        --accent-color: #4CAF50;
                        --card-bg: #2d2d2d;
                        --border-color: #404040;
                    }
                    
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                        background-color: var(--bg-color);
                        color: var(--text-color);
                        line-height: 1.6;
                        padding: 20px;
                        min-height: 100vh;
                    }
                    
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    
                    header {
                        text-align: center;
                        margin-bottom: 2rem;
                    }
                    
                    h1 {
                        font-size: 2.5rem;
                        margin-bottom: 0.5rem;
                        color: var(--accent-color);
                    }
                    
                    .subtitle {
                        color: #888;
                        font-size: 1.1rem;
                    }
                    
                    .stream-container {
                        background: var(--card-bg);
                        border-radius: 12px;
                        padding: 20px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin: 20px auto;
                        border: 1px solid var(--border-color);
                    }
                    
                    .stream {
                        width: 100%;
                        max-width: 100%;
                        height: auto;
                        border-radius: 8px;
                        display: block;
                    }
                    
                    .info-panel {
                        background: var(--card-bg);
                        border-radius: 12px;
                        padding: 20px;
                        margin-top: 20px;
                        border: 1px solid var(--border-color);
                    }
                    
                    .legend {
                        display: flex;
                        gap: 20px;
                        flex-wrap: wrap;
                        justify-content: center;
                        margin-top: 15px;
                    }
                    
                    .legend-item {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    
                    .legend-color {
                        width: 20px;
                        height: 20px;
                        border-radius: 4px;
                    }
                    
                    .green { background-color: #4CAF50; }
                    .yellow { background-color: #FFD700; }
                    .orange { background-color: #FFA500; }
                    
                    @media (max-width: 768px) {
                        body {
                            padding: 10px;
                        }
                        
                        .container {
                            padding: 10px;
                        }
                        
                        h1 {
                            font-size: 2rem;
                        }
                        
                        .stream-container {
                            padding: 10px;
                        }
                    }
                    
                    @media (prefers-color-scheme: light) {
                        :root {
                            --bg-color: #f5f5f5;
                            --text-color: #333333;
                            --card-bg: #ffffff;
                            --border-color: #e0e0e0;
                        }
                        
                        .subtitle {
                            color: #666;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <header>
                        <h1>Is it poppin' ?</h1>
                        <p class="subtitle">TensorFlow Powered Person Counter</p>
                    </header>
                    
                    <div class="stream-container">
                        <img src="/video_feed" class="stream" alt="Camera Feed" />
                    </div>
                    
                    <div class="info-panel">
                        <h2>Detection Information</h2>
                        <div class="legend">
                            <div class="legend-item">
                                <div class="legend-color green"></div>
                                <span>High Confidence (>70%)</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color yellow"></div>
                                <span>Medium Confidence (50-70%)</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color orange"></div>
                                <span>Low Confidence (<50%)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Insert the model switcher before the closing body tag
            template = template.replace('</body>', model_switcher_html + '</body>')
            # Add the new styles to the existing styles
            template = template.replace('</style>', additional_styles + '</style>')
            
            # Render the template with the model data
            return render_template_string(
                template,
                models=models,
                current_model=current_model
            )

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.get_frame(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

    def run(self, host='0.0.0.0', port=5000):
        # Start capture thread
        thread = threading.Thread(target=self.capture_frames)
        thread.daemon = True
        thread.start()
        
        # Run Flask app
        self.app.run(host=host, port=port)

    def __del__(self):
        self.running = False 