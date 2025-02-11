# People Counter

A Python application that counts people entering and exiting using RTSPS camera feed and Coral AI.

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python run.py
```

Open your web browser and navigate to:

Access the web interface at: `http://your_server_ip:5000`

## Requirements

- Python 3.8+
- OpenCV
- Flask
- Coral TPU