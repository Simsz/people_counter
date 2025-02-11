from flask import Flask, Response, render_template_string, request
import cv2
import time
import os
from datetime import datetime
from .detector import PersonDetector

class CameraStream:
    def __init__(self, camera_url, detector=None):
        self.camera_url = camera_url
        self.detector = detector
        self.app = Flask(__name__)
        self.width = int(os.getenv('STREAM_WIDTH', '1280'))
        self.height = int(os.getenv('STREAM_HEIGHT', '720'))
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.setup_routes()
        
    def generate_frames(self):
        while True:
            try:
                print("Attempting to connect to camera...")
                cap = cv2.VideoCapture(self.camera_url)
                
                if not cap.isOpened():
                    print("Failed to open camera. Retrying...")
                    time.sleep(5)
                    continue
                
                while True:
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame. Reconnecting...")
                        break
                    
                    # Calculate FPS
                    self.frame_count += 1
                    if time.time() - self.last_fps_time > 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_fps_time = time.time()
                    
                    # Resize frame
                    frame = cv2.resize(frame, (self.width, self.height))
                    
                    # Run detection
                    if self.detector:
                        try:
                            detections, tracks = self.detector.detect(frame)
                            frame = self.detector.draw_detections(frame, (detections, tracks))
                            
                            # Add performance overlay
                            process_time = time.time() - start_time
                            cv2.putText(frame, f'FPS: {self.fps}', (10, 70), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f'Process Time: {process_time:.3f}s', (10, 110), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Log detections
                            if len(detections) > 0:
                                print(f"[{datetime.now()}] Detected {len(detections)} people")
                                
                        except Exception as e:
                            print(f"Detection error: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
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
                if 'cap' in locals():
                    cap.release()

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
                        <h1>People Counter</h1>
                        <p class="subtitle">AI-Powered People Detection</p>
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
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port) 