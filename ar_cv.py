import cv2
import numpy as np
from flask import Flask, render_template, Response
import mido
import json

app = Flask(__name__)
phone_camera_url = 'http://192.168.101.15:8080/video'
camera = cv2.VideoCapture(phone_camera_url)
color_to_track = np.array([30, 150, 50])
piano_keys = ["C", "D", "E", "F", "G", "A", "B", "C"]

prev_detected_keys = set()

with open('piano_key_mapping.json', 'r') as json_file:
    midi_to_piano_keys = json.load(json_file)

def map_keys(x, w):
    key_index = int((x + w / 2) * len(piano_keys) / 640) % len(piano_keys)
    return piano_keys[key_index]

def generate_frames():
    global prev_detected_keys

    with mido.open_input('Digital Keyboard 0') as inport:
        while True:
            success, frame = camera.read()
            if not success:
                text_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(text_frame, 'No input available', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', text_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            else:
                frame = cv2.resize(frame, (640, 480))
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_frame, np.array([20, 100, 100]), np.array([40, 255, 255]))
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                current_detected_keys = set()

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    key = map_keys(x, w)
                    cv2.putText(frame, key, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    current_detected_keys.add(key)

                gone_keys = prev_detected_keys - current_detected_keys
                #if gone_keys:
                    #print(f"The following keys are entirely gone: {', '.join(gone_keys)}")

                prev_detected_keys = current_detected_keys

                for msg in inport.iter_pending():
                    if msg.type == 'note_on':
                        note = str(msg.note)
                        velocity = str(msg.velocity)
                        key = midi_to_piano_keys.get(note, "Unknown Key")
                        if msg.velocity == 0:
                            print(f"Released: {note}, Velocity: {velocity}, Key: {key}")
                        else:
                            print(f"Pressed: {note}, Velocity: {velocity}, Key: {key}")


                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
