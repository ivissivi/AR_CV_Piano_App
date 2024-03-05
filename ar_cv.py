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
        self.key_pressed = {key: False for key in piano_keys}
        self.song_index = 0  # Track the index of the current key in the song
        self.red_line_start = None  # Track the start position for the red line

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

        for key in piano_keys:
            red_line_start = None  # Reset red line for each key
            first_occurrence = True  # Track the first occurrence of the key
            for index, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                current_key = piano_keys[index % self.num_keys]

                self.key_positions[current_key] = (x, y)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text_background_height = 25
                cv2.rectangle(frame, (x, y - text_background_height), (x + w, y), (0, 0, 0), -1)
                cv2.putText(frame, current_key, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if current_key == key and not self.key_pressed[key]:
                    if first_occurrence:
                        red_line_start = (x + w // 2, y)
                        first_occurrence = False

            if red_line_start is not None and key == song[self.song_index]:
                self.red_line_start = red_line_start
                red_line_end = (red_line_start[0], self.frame_size[1])
                cv2.line(frame, red_line_start, red_line_end, (0, 0, 255), 2)

        return frame

    def move_to_next_key(self):
        if self.song_index < len(song) - 1:
            self.song_index += 1
        else:
            # Reset to the beginning of the song if the end is reached
            self.song_index = 0

    def generate_frames(self):
        with mido.open_input('Digital Keyboard 0') as inport:
            while True:
                success, frame = camera.read()
                if not success:
                    text_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(text_frame, 'No input available', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    ret, buffer = cv2.imencode('.jpg', text_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                else:
                    frame = self.detect_and_outline_keys(frame)

                    for msg in inport.iter_pending():
                        if msg.type == 'note_on':
                            note = str(msg.note)
                            velocity = str(msg.velocity)
                            key = midi_to_piano_keys.get(note, "Unknown Key")
                            if msg.velocity == 0:
                                recognition.key_pressed[key] = False
                                print(f"Released: {note}, Velocity: {velocity}, Key: {key}")
                            else:
                                recognition.key_pressed[key] = True
                                print(f"Pressed: {note}, Velocity: {velocity}, Key: {key}")
                                if key == song[self.song_index]:
                                    # Move to the next key in the song
                                    self.move_to_next_key()

                    frame = cv2.GaussianBlur(frame, (1, 1), 0)
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

recognition = PianoKeyRecognition()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(recognition.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
