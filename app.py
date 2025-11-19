import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO

# Flask app ko initialize karo
app = Flask(__name__)

# Aapka trained model load karo
# Path ko apne project ke hisaab se set karo
model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)

def generate_frames():
    
    camera = cv2.VideoCapture(0)  # Default webcam
    if not camera.isOpened():
        print("Error: Webcam is not working.")
        return

    while True:
        # Frame capture karo
        success, frame = camera.read()
        if not success:
            print("Error: Frame is not captured .")
            break
        else:
            # Frame par YOLOv8 detection run karo
            # stream=True efficient hai streaming ke liye
            results = model(frame, stream=True)

            # Results ko iterate karo aur frame par plot karo
            for r in results:
                annotated_frame = r.plot() # Yeh boxes aur labels draw karega

                # Frame ko JPEG format mein encode karo
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    continue
                
                # Frame ko bytes mein convert karo
                frame_bytes = buffer.tobytes()

                # Frame ko stream karo
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def login():
    #handels login pg
    error = None
    if request.method == 'POST':
        
        if request.form['username'] == 'sidd' and request.form['password'] == 'password':
            
            return redirect(url_for('feed_page'))
        else:
            
            error = 'Invalid Credentials. Please try again.'
            
    
    return render_template('login.html', error=error)

@app.route('/feed')
def feed_page():
    """Video feed wala page render karo."""
    return render_template('feed.html')

@app.route('/video_stream')
def video_stream():
    """Video stream ko feed.html page par bhejo."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# App ko run karo
if __name__ == '__main__':
    app.run(debug=True)