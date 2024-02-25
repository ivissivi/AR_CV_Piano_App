import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

phone_camera_url = 'http://192.168.101.15:8080/video'
camera = cv2.VideoCapture(phone_camera_url)

color_to_track = np.array([30, 150, 50])

piano_keys = ["C", "D", "E", "F", "G", "A", "B", "C"]

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            text_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(text_frame, 'No input available', (50, 240), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', text_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
        else:
            frame = cv2.resize(frame, (640, 480))
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([20, 100, 100])
            upper_bound = np.array([40, 255, 255])
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for idx, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                key_index = int((x + w / 2) / (640 / len(piano_keys)))
                cv2.putText(frame, piano_keys[key_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
