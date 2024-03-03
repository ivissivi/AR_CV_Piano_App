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
        self.key_positions = {}

    def detect_and_outline_keys(self, frame):
        frame = cv2.resize(frame, (640, 480))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        color_range = (np.array([20, 100, 100]), np.array([40, 255, 255]))

        combined_mask = cv2.inRange(hsv_frame, color_range[0], color_range[1])

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_keys = set()

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Calculate relative position of the key
                relative_position = cx / frame.shape[1]

                # Find the closest key based on relative position
                key_index = min(len(piano_keys) - 1, max(0, int(round(relative_position * (len(piano_keys) - 1)))))

                key = piano_keys[key_index]
                detected_keys.add(key)

                text = piano_keys[key_index]
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]  # Increase text size
                text_position = (cx - text_size[0] // 2, cy + text_size[1] // 2)  # Center text

                cv2.rectangle(frame, (text_position[0] - 5, text_position[1] - 5 - text_size[1]),
                              (text_position[0] - 5 + text_size[0] + 10, text_position[1] + 5),
                              (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if text in self.key_positions:
                    line_start = (cx, cy)
                    line_end = (cx, frame.shape[0])
                    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

                self.key_positions[text] = (cx, cy)

        if "C" in detected_keys:
            detected_keys = {"C"}

        detected_keys = sorted(detected_keys, key=lambda x: self.key_positions.get(x, (0, 0))[0])

        new_keys = set(detected_keys) - set(self.prev_detected_keys)
        for new_key in new_keys:
            print(f"Key {new_key} detected")

        # Print the current order of keys
        print("Current order of keys:", detected_keys)

        self.prev_detected_keys = detected_keys

        return frame

recognition = PianoKeyRecognition()

def generate_frames():
    with mido.open_input('Digital Keyboard 0') as inport:
        while True:
            success, frame = camera.read()

            if not success:
                text_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(text_frame, 'No input available', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', text_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            else:
                frame = recognition.detect_and_outline_keys(frame)
                frame = cv2.GaussianBlur(frame, (1, 1), 0)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

                note = "Unknown Note"

                for msg in inport.iter_pending():
                    if msg.type == 'note_on':
                        note = str(msg.note)
                        key = midi_to_piano_keys.get(note, "Unknown Key")
                        velocity = str(msg.velocity)

                        print(f"{'Released' if msg.velocity == 0 else 'Pressed'}: Note: {note}, Velocity: {velocity}, Key: {key}")

                        if key in recognition.key_positions:
                            print(f"Removing red line for Key: {key}")
                            recognition.key_positions.pop(key)
                        else:
                            print(f"No red line found for Key: {key}. Existing keys: {list(recognition.key_positions.keys())}")

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
