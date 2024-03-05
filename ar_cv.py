import cv2
import numpy as np
from flask import Flask, render_template, Response
import mido
import json

app = Flask(__name__)
phone_camera_url = 'http://192.168.101.15:8080/video'
camera = cv2.VideoCapture(phone_camera_url)
piano_keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C"]

song = ["C", "D", "D", "C", "A", "F#", "G", "B", "A", "A#", "B", "C"]

with open('piano_key_mapping.json', 'r') as json_file:
    midi_to_piano_keys = json.load(json_file)

class PianoKeyRecognition:
    def __init__(self, frame_size=(640, 480), num_keys=13):
        self.frame_size = frame_size
        self.num_keys = num_keys
        self.key_positions = {}

    def detect_and_outline_keys(self, frame):
        frame = cv2.resize(frame, self.frame_size)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, np.array([20, 100, 100]), np.array([40, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_width = sum(cv2.boundingRect(contour)[2] for contour in contours)
        average_width = total_width / len(contours) if len(contours) > 0 else 1

        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        red_line_start = None

        for index, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            key = piano_keys[index % self.num_keys]

            self.key_positions[key] = (x, y)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text_background_height = 25
            cv2.rectangle(frame, (x, y - text_background_height), (x + w, y), (0, 0, 0), -1)
            cv2.putText(frame, key, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if key == "C":
                red_line_start = (x + w // 2, y)

        if red_line_start is not None:
            red_line_end = (red_line_start[0], self.frame_size[1])
            cv2.line(frame, red_line_start, red_line_end, (0, 0, 255), 2)

        return frame

recognition = PianoKeyRecognition()

def generate_frames():
    with mido.open_input('Digital Keyboard 0') as inport:
        while True:
            success, frame = camera.read()
            if not success:
                text_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(text_frame, 'No input available', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', text_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            else:
                frame = recognition.detect_and_outline_keys(frame)

                for msg in inport.iter_pending():
                    if msg.type == 'note_on':
                        note = str(msg.note)
                        velocity = str(msg.velocity)
                        key = midi_to_piano_keys.get(note, "Unknown Key")
                        if msg.velocity == 0:
                            print(f"Released: {note}, Velocity: {velocity}, Key: {key}")
                        else:
                            print(f"Pressed: {note}, Velocity: {velocity}, Key: {key}")

                frame = cv2.GaussianBlur(frame, (1, 1), 0)
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
