import cv2
import numpy as np
from flask import Flask, render_template, Response
import mido
import json

app = Flask(__name__)

phone_camera_url = 'http://192.168.101.15:8080/video'
camera = cv2.VideoCapture(phone_camera_url)

with open('piano_key_mapping.json', 'r') as json_file:
    midi_to_piano_keys = json.load(json_file)

piano_keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C"]

class PianoKeyRecognition:
    def __init__(self):
        self.prev_detected_keys = set()

    def detect_and_outline_keys(self, frame):
        frame = cv2.resize(frame, (640, 480))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        color_ranges = [
            (np.array([20, 100, 100]), np.array([40, 255, 255]))
        ]

        combined_mask = np.zeros_like(frame[:, :, 0])
        for lower, upper in color_ranges:
            combined_mask |= cv2.inRange(hsv_frame, lower, upper)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours (noise)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 50]

        detected_keys = set()

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                key_index = int((cx / 640) * len(piano_keys))

                if 0 <= key_index < len(piano_keys):
                    detected_keys.add(piano_keys[key_index])

                    text = piano_keys[key_index]
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(frame, (cx - 5, cy - 5 - text_size[1]), (cx - 5 + text_size[0], cy - 5), (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, text, (cx - 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Print the number of rectangles detected
        print(f"Number of rectangles detected: {len(contours)}")

        # Reorder the detected keys based on the centroid y-coordinate
        detected_keys = sorted(detected_keys, key=lambda x: cv2.moments(contours[piano_keys.index(x)])["m01"])

        new_keys = set(detected_keys) - set(self.prev_detected_keys)
        for new_key in new_keys:
            print(f"Key {new_key} detected")

        self.prev_detected_keys = detected_keys

        return frame

recognition = PianoKeyRecognition()

def generate_frames():
    with mido.open_input('Digital Keyboard 0') as inport:
        while True:
            success, frame = camera.read()

            if not success:
                text_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(text_frame, 'No input available', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', text_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            else:
                frame = recognition.detect_and_outline_keys(frame)
                frame = cv2.GaussianBlur(frame, (1, 1), 0)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

                for msg in inport.iter_pending():
                    if msg.type == 'note_on':
                        note, velocity, key = str(msg.note), str(msg.velocity), midi_to_piano_keys.get(str(msg.note), "Unknown Key")
                        print(f"{'Released' if msg.velocity == 0 else 'Pressed'}: {note}, Velocity: {velocity}, Key: {key}")

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
