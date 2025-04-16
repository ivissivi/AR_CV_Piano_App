import cv2
import numpy as np
from flask import Flask, render_template, Response, request, send_file
import mido
import json
import pygame
import os
import argparse
import qrcode
import socket
import io
from PIL import Image
import base64
import threading

app = Flask(__name__)

# Configuration
DEFAULT_PORT = 5000
DEFAULT_HOST = '0.0.0.0'

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

def generate_qr_code(url):
    """Generate a QR code for the given URL"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

# Load piano key mapping
with open('piano_key_mapping.json', 'r') as json_file:
    midi_to_piano_keys = json.load(json_file)

# Initialize Pygame mixer
pygame.mixer.init()

# Sample song (can be loaded from file in future)
song = ["C", "C", "D", "C", "F", "E", "C", "C", "D", "C", "G", "F", "C", "C", "C1", "A", "F", "F", "E", "D", "A#", "A#", "A", "F", "G", "F"]
piano_keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C1"]

class PianoKeyRecognition:
    def __init__(self, frame_size=(640, 480), num_keys=13):
        self.frame_size = frame_size
        self.num_keys = num_keys
        self.key_positions = {}
        self.key_pressed = {key: False for key in piano_keys}
        self.song_index = 0
        self.red_line_start = None
        self.midi_port = None
        self.qr_code = None
        self.current_frame = None
        self.frame_lock = threading.Lock()

    def initialize_midi(self):
        """Initialize MIDI input"""
        try:
            self.midi_port = mido.open_input('Digital Keyboard 0')
            return True
        except:
            print("Warning: Could not connect to MIDI keyboard. Running in visual-only mode.")
            return False

    def process_frame(self, frame):
        """Process a single frame from the camera"""
        with self.frame_lock:
            self.current_frame = frame

    def detect_and_outline_keys(self, frame):
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame = cv2.resize(frame, self.frame_size)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow color detection (for piano keys)
        mask = cv2.inRange(hsv_frame, np.array([20, 100, 100]), np.array([40, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return frame

        total_width = sum(cv2.boundingRect(contour)[2] for contour in contours)
        average_width = total_width / len(contours)

        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        for key in piano_keys:
            red_line_start = None
            first_occurrence = True
            for index, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                current_key = piano_keys[index % self.num_keys]

                self.key_positions[current_key] = (x, y)

                # Draw key outline
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw key label
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
                                note_sound = pygame.mixer.Sound(f"notes/{key}.wav")
                                note_sound.play()
                                if key == song[self.song_index]:
                                    self.move_to_next_key()

                    frame = cv2.GaussianBlur(frame, (1, 1), 0)
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Initialize recognition system
recognition = PianoKeyRecognition()

@app.route('/')
def index():
    return render_template('index.html', qr_code=recognition.qr_code is not None)

@app.route('/video_feed')
def video_feed():
    return Response(recognition.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/qr_code')
def qr_code():
    return Response(recognition.qr_code, mimetype='image/png')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if 'frame' not in request.files:
        return 'No frame uploaded', 400
    
    file = request.files['frame']
    if file.filename == '':
        return 'No selected file', 400
    
    # Read the image file
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the frame
    recognition.process_frame(frame)
    
    return 'Frame received', 200

def main():
    parser = argparse.ArgumentParser(description='AR Piano Learning Application')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help='Web server port')
    parser.add_argument('--host', type=str, default=DEFAULT_HOST,
                        help='Web server host')
    
    args = parser.parse_args()

    try:
        print("Initializing AR Piano Learning Application...")
        print("Press Ctrl+C to stop the application")
        
        midi_connected = recognition.initialize_midi()
        
        # Generate QR code
        local_ip = get_local_ip()
        recognition.generate_qr_code(local_ip, args.port)
        
        print(f"\nApplication is running!")
        print(f"Web interface: http://{local_ip}:{args.port}")
        print(f"MIDI keyboard: {'Connected' if midi_connected else 'Not connected (running in visual-only mode)'}")
        print("\nScan the QR code on your phone to connect!")
        print("\nInstructions:")
        print("1. Open the web interface on your phone")
        print("2. Grant camera permissions when prompted")
        print("3. Click 'Start Camera' to begin")
        print("4. Position your phone to view the piano keys")
        
        app.run(host=args.host, port=args.port, debug=True)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure your phone and computer are on the same network")
        print("2. Check if the port is available")
        print("3. Try restarting the application")
    finally:
        if recognition.midi_port:
            recognition.midi_port.close()

if __name__ == '__main__':
    main()
