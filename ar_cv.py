import cv2
import numpy as np
from flask import Flask, render_template, Response, request, send_file, jsonify, send_from_directory
import mido
import json
import os
import argparse
import qrcode
import socket
import io
from PIL import Image
import base64
import threading
import time
import sys
import ssl
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

DEFAULT_PORT = 5000
DEFAULT_HOST = '0.0.0.0'

mido.set_backend('mido.backends.rtmidi')

MIN_KEY_WIDTH = 20
MAX_KEY_WIDTH = 100
MIN_KEY_HEIGHT = 50
KEY_COLOR_THRESHOLD = 100
BLACK_KEY_RATIO = 0.6

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

def generate_qr_code(url):
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=4,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        return qr
    except Exception as e:
        print(f"❌ Error generating QR code: {str(e)}")
        return None

def print_qr_code(qr, label):
    try:
        print(f"\n📱 {label} QR Code:")
        width = qr.modules_count + 2
        # Top border
        print("┌" + "─" * width + "┐")
        modules = qr.modules
        n = len(modules)
        # Print two QR rows per terminal row
        for y in range(0, n, 2):
            print("│", end="")
            for x in range(n):
                top = modules[y][x]
                bottom = modules[y+1][x] if y+1 < n else False
                if top and bottom:
                    char = "█"
                elif top and not bottom:
                    char = "▀"
                elif not top and bottom:
                    char = "▄"
                else:
                    char = " "
                print(char, end="")
            print("│")
        # Bottom border
        print("└" + "─" * width + "┘")
    except Exception as e:
        print(f"❌ Error printing QR code: {str(e)}")

with open('piano_key_mapping.json', 'r') as json_file:
    midi_to_piano_keys = json.load(json_file)

song = ["C", "C", "D", "C", "F", "E", "C", "C", "D", "C", "G", "F", "C", "C", "C1", "A", "F", "F", "E", "D", "A#", "A#", "A", "F", "G", "F"]
piano_keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C1"]

class PianoKeyRecognition:
    def __init__(self, frame_size=(640, 480), num_keys=13):
        self.frame_size = frame_size
        self.num_keys = num_keys
        self.camera = None
        self.midi_input = None
        self.midi_output = None
        self.key_positions = []
        self.piano_learning_mode = False
        self.current_target_note = None
        self.note_sequence = ['C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2', 'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3', 
                            'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5',
                            'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6', 'C7']
        self.current_note_index = 0
        self.correct_notes = 0
        self.total_notes = 0
        self.note_start_time = None
        self.note_duration = 2.0
        self.key_pressed = {key: False for key in piano_keys}
        self.song_index = 0
        self.red_line_start = None
        self.qr_code = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_available = threading.Event()
        self.midi_thread = None
        self.midi_running = False
        self.camera_error = None
        self.connected_clients = set()

    def initialize_camera(self):
        try:
            print("\n📸 Initializing camera...")
            if self.camera is not None:
                self.camera.release()
            
            for i in range(3):
                self.camera = cv2.VideoCapture(i)
                if self.camera.isOpened():
                    print(f"✅ Camera {i} opened successfully")
                    break
                else:
                    self.camera.release()
            
            if not self.camera.isOpened():
                raise Exception("No camera could be opened")
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            success, frame = self.camera.read()
            if not success:
                raise Exception("Could not read from camera")
            
            print("✅ Camera initialized successfully")
            self.camera_error = None
            return True
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Camera Error: {error_msg}")
            self.camera_error = error_msg
            return False

    def midi_listener(self):
        try:
            print("\n🎹 === Starting MIDI Listener Thread ===")
            mido.set_backend('mido.backends.rtmidi')
            time.sleep(1)
            
            available_ports = mido.get_input_names()
            print(f"\n🔍 Available MIDI ports in thread:")
            for port in available_ports:
                print(f"   • {port}")
            
            if not available_ports:
                print("\n❌ No MIDI ports found in thread")
                return
            
            target_port = "Digital Keyboard 0"
            if target_port in available_ports:
                print(f"\n🔌 Attempting to connect to {target_port}...")
                try:
                    with mido.open_input(target_port) as port:
                        print(f"\n✅ Successfully connected to {target_port}")
                        self.midi_input = port
                        self.midi_running = True
                        
                        print("\n🎵 Ready to receive MIDI messages!")
                        print("   Press keys on your MIDI keyboard to test")
                        print("   Press Ctrl+C to stop")
                        
                        while self.midi_running:
                            for msg in port.iter_pending():
                                if msg.type == 'clock':
                                    continue
                                    
                                print(f"\n📥 Raw MIDI message: {msg}")
                                self.process_midi_message(msg)
                except Exception as e:
                    print(f"\n❌ Error in MIDI thread: {str(e)}")
                    print(f"Error details: {str(e)}")
            else:
                print(f"\n❌ Could not find {target_port} in available ports")
        except Exception as e:
            print(f"\n❌ MIDI thread error: {str(e)}")
            print(f"Error details: {str(e)}")
        finally:
            self.midi_running = False
            self.midi_input = None

    def initialize_midi(self):
        try:
            if self.midi_thread is None or not self.midi_thread.is_alive():
                self.midi_running = True
                self.midi_thread = threading.Thread(target=self.midi_listener)
                self.midi_thread.daemon = True
                self.midi_thread.start()
                print("\n🔄 Starting MIDI listener thread...")
                return True
            return False
        except Exception as e:
            print(f"\n❌ Error starting MIDI thread: {str(e)}")
            return False

    def process_frame(self, frame):
        try:
            print("[Frame Debug] Entering process_frame")
            with self.frame_lock:
                print("[Frame Debug] Acquired frame lock")
                if frame is not None:
                    print(f"[Frame Debug] Processing frame: {frame.shape}")
                    self.current_frame = frame.copy()
                    print(f"[Frame Debug] Made copy of frame: {self.current_frame.shape}")
                    processed_frame = self.detect_piano_keys(frame)
                    print(f"[Frame Debug] Detected piano keys in frame")
                    self.frame_available.set()
                    print(f"[Frame Debug] Set frame_available event")
                    return processed_frame
                else:
                    print("[Frame Debug] Received None frame in process_frame")
                    self.frame_available.clear()
                    return None
        except Exception as e:
            print(f"[Frame Debug] Error in process_frame: {str(e)}")
            self.frame_available.clear()
            return None
        finally:
            print("[Frame Debug] Exiting process_frame")

    def detect_piano_keys(self, frame):
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        
        print(f"\n[Piano Detection] Starting detection on frame: {width}x{height}")
        
        if not hasattr(self, 'calibration_points') or len(self.calibration_points) != 2:
            print("[Piano Detection] ❌ Calibration points not set or incomplete")
            print(f"[Piano Detection] Current calibration points: {getattr(self, 'calibration_points', None)}")
            cv2.putText(output_frame, "Please calibrate first", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return output_frame
            
        print(f"[Piano Detection] ✅ Calibration points found: {self.calibration_points}")
        
        left_point = self.calibration_points[0]
        right_point = self.calibration_points[1]
        
        print(f"[Piano Detection] Left point: {left_point}, Right point: {right_point}")
        
        # Calculate key spacing based on the actual distance between calibration points
        total_distance = right_point['x'] - left_point['x']
        key_spacing = total_distance / 12  # Divide by 12 to get spacing for one octave
        
        # Generate keys starting from the left calibration point
        self.key_positions = []
        current_x = left_point['x']
        
        # Generate 5 octaves of keys (60 keys total)
        for octave in range(5):  # 5 octaves
            for note in range(12):  # 12 notes per octave
                midi_note = 36 + (octave * 12) + note  # Start from C2 (MIDI 36)
                if midi_note > 108:  # Stop at C7 (MIDI 108)
                    break
                    
                key_info = {
                    'x': int(current_x),
                    'y': int(left_point['y']),
                    'width': int(key_spacing),
                    'height': int(height * 0.6),
                    'is_white': self.is_white_key(midi_note),
                    'note': midi_note,
                    'note_name': self.get_note_name(midi_note)
                }
                self.key_positions.append(key_info)
                
                # Draw the key
                if key_info['is_white']:
                    cv2.rectangle(output_frame, 
                                (int(current_x), key_info['y']),
                                (int(current_x + key_spacing), key_info['y'] + key_info['height']),
                                (255, 255, 255), -1)
                    cv2.rectangle(output_frame,
                                (int(current_x), key_info['y']),
                                (int(current_x + key_spacing), key_info['y'] + key_info['height']),
                                (0, 0, 0), 1)
                else:
                    cv2.rectangle(output_frame,
                                (int(current_x), key_info['y']),
                                (int(current_x + key_spacing), key_info['y'] + key_info['height']),
                                (0, 0, 0), -1)
                    cv2.rectangle(output_frame,
                                (int(current_x), key_info['y']),
                                (int(current_x + key_spacing), key_info['y'] + key_info['height']),
                                (255, 255, 255), 1)
                
                current_x += key_spacing
        
        print(f"[Piano Detection] ✅ Generated {len(self.key_positions)} keys with spacing {key_spacing:.2f} pixels")
        return output_frame

    def is_white_key(self, midi_note):
        # White keys are C, D, E, F, G, A, B
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = note_names[midi_note % 12]
        return note_name in ['C', 'D', 'E', 'F', 'G', 'A', 'B']

    def generate_frames(self):
        if not self.initialize_camera():
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, 'Camera Error', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_frame, 'Please check permissions', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(error_frame, 'and try again', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return

        while True:
            try:
                success, frame = self.camera.read()
                if not success:
                    print("\n❌ Error reading from camera")
                    break

                frame = cv2.resize(frame, (640, 480))
                
                processed_frame = self.detect_piano_keys(frame)
                
                if len(processed_frame.shape) == 2:
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"\n❌ Frame generation error: {str(e)}")
                break

    def __del__(self):
        print("\n🛑 Cleaning up resources...")
        self.midi_running = False
        if self.midi_thread and self.midi_thread.is_alive():
            self.midi_thread.join(timeout=1.0)
        if self.midi_input:
            try:
                self.midi_input.close()
                print("✅ MIDI input port closed successfully")
            except:
                print("❌ Error closing MIDI input port")
        if self.camera:
            try:
                self.camera.release()
                print("✅ Camera released successfully")
            except:
                print("❌ Error releasing camera")

    def process_midi_message(self, msg):
        if msg.type == 'note_on' and msg.velocity > 0:
            note_name = self.get_note_name(msg.note)
            
            socketio.emit('midi', {
                'status': 'note_on',
                'note': msg.note,
                'velocity': msg.velocity,
                'note_name': note_name
            })
            
            if self.piano_learning_mode and self.current_target_note:
                if note_name == self.current_target_note:
                    socketio.emit('learning_feedback', {
                        'correct': True,
                        'message': 'Correct!',
                        'next_note': self.get_next_note()
                    })
                    self.correct_notes += 1
                    self.total_notes += 1
                    self.current_note_index = (self.current_note_index + 1) % len(self.note_sequence)
                    self.current_target_note = self.note_sequence[self.current_note_index]
                else:
                    socketio.emit('learning_feedback', {
                        'correct': False,
                        'message': f'Wrong note! Try {self.current_target_note}',
                        'next_note': self.current_target_note
                    })
            
            self.highlight_key(note_name)

    def get_note_name(self, midi_note):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # Adjust for C2 (MIDI 36) starting point
        octave = ((midi_note - 36) // 12) + 2
        note = notes[midi_note % 12]
        return f"{note}{octave}"

    def highlight_key(self, note_name):
        for key in self.key_positions:
            if key['note_name'] == note_name:
                highlight = {
                    'x': key['x'],
                    'y': key['y'],
                    'width': key['width'],
                    'height': key['height'],
                    'color': (0, 255, 255) if key['is_white'] else (255, 255, 0),
                    'opacity': 0.7,
                    'blur': 5,
                    'duration': 200,
                    'note_name': note_name
                }
                print(f"[Key Highlight] Highlighting key: {note_name} at position ({key['x']}, {key['y']})")
                socketio.emit('key_highlight', highlight)
                break

    def start_piano_learning_mode(self):
        self.piano_learning_mode = True
        self.current_note_index = 0
        self.correct_notes = 0
        self.total_notes = 0
        self.current_target_note = self.note_sequence[0]
        socketio.emit('start_learning', {
            'current_note': self.current_target_note,
            'sequence': self.note_sequence
        })

    def stop_piano_learning_mode(self):
        self.piano_learning_mode = False
        self.current_target_note = None
        socketio.emit('stop_learning')

    def get_next_note(self):
        next_index = (self.current_note_index + 1) % len(self.note_sequence)
        return self.note_sequence[next_index]

# Initialize recognition system
recognition = PianoKeyRecognition()

# Always generate QR code for the fixed URL
fixed_url = 'https://192.168.8.11:8000'
qr = generate_qr_code(fixed_url)
if qr:
    # Convert QR code to PNG base64
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    recognition.qr_code = base64.b64encode(buffered.getvalue()).decode('utf-8')
else:
    recognition.qr_code = None

# Video stream class to manage frames and clients
class VideoStream:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 15

    def update(self, frame):
        current_time = time.time()
        with self.lock:
            if current_time - self.last_frame_time >= self.frame_interval:
                self.frame = frame
                self.last_frame_time = current_time

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

video_stream = VideoStream()

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    recognition.connected_clients.add(client_id)
    print(f'[Socket Debug] Client connected: {client_id}')
    print(f'[Socket Debug] Total connected clients: {len(recognition.connected_clients)}')
    emit('status', {'message': 'Connected to server'})
    
    # Send QR code to desktop client
    if request.headers.get('User-Agent', '').find('Mobile') == -1:
        # This is a desktop client
        if recognition.qr_code:
            emit('qr_code', recognition.qr_code)
            print(f'[Socket Debug] Sent QR code to desktop client: {client_id}')

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    recognition.connected_clients.discard(client_id)
    print(f'[Socket Debug] Client disconnected: {client_id}')
    print(f'[Socket Debug] Total connected clients: {len(recognition.connected_clients)}')

@socketio.on('log')
def handle_log(data):
    print(f"[{data['timestamp']}] {data['message']}")

@socketio.on('frame')
def handle_frame(data):
    try:
        client_id = request.sid
        print(f"[Socket Debug] Received frame from client: {client_id}")
        
        # Decode the base64 image data
        image_data = base64.b64decode(data)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("[Frame Error] Failed to decode frame")
            return
            
        # Resize frame if needed
        height, width = frame.shape[:2]
        if width > 640 or height > 480:
            frame = cv2.resize(frame, (640, 480))
            
        # Process the frame
        processed_frame = recognition.process_frame(frame)
        
        if processed_frame is not None:
            # Encode the processed frame
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                print("[Frame Error] Failed to encode processed frame")
                return
                
            # Convert to base64 and send to all clients
            frame_data = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('frame', frame_data, broadcast=True)
            print("[Frame Debug] Frame broadcasted to clients")
        else:
            print("[Frame Error] Processed frame is None")
            
    except Exception as e:
        print(f"[Frame Error] Error processing frame: {str(e)}")
        print(f"[Frame Error] Error type: {type(e)}")
        print(f"[Frame Error] Error details: {str(e)}")
        socketio.emit('error', {'message': 'Error processing frame'}, room=client_id)

@app.route('/')
def index():
    return render_template('index.html', 
                         qr_code=recognition.qr_code is not None,
                         camera_error=recognition.camera_error)

@app.route('/desktop')
def desktop():
    try:
        print("\n🖥️ Desktop view requested")
        return render_template('desktop.html')
    except Exception as e:
        print(f"\n❌ Error loading desktop view: {str(e)}")
        return "Error loading desktop view", 500

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        frame_data = request.files['frame'].read()
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        video_stream.update(frame)
        return {'status': 'success'}
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {'status': 'error', 'message': str(e)}

@app.route('/start_camera', methods=['POST'])
def start_camera():
    if recognition.initialize_camera():
        return {'status': 'success', 'message': 'Camera started successfully'}
    else:
        return {'status': 'error', 'message': recognition.camera_error or 'Unknown camera error'}

@app.route('/get_songs')
def get_songs():
    try:
        print("\n🎵 === Song Selection Request ===")
        songs_dir = os.path.join(app.root_path, 'songs')
        songs = []
        print("🔍 Scanning songs directory:", songs_dir)
        
        for filename in os.listdir(songs_dir):
            if filename.endswith('.json'):
                song_path = os.path.join(songs_dir, filename)
                print(f"📝 Found song file: {filename}")
                try:
                    with open(song_path, 'r') as f:
                        song_data = json.load(f)
                        songs.append({
                            'title': song_data.get('title', filename),
                            'filename': filename,
                            'difficulty': song_data.get('difficulty', 'unknown'),
                            'tempo': song_data.get('tempo', 0),
                            'time_signature': song_data.get('time_signature', '4/4'),
                            'key': song_data.get('key', 'C major')
                        })
                        print(f"✅ Loaded song: {song_data.get('title', filename)}")
                except Exception as e:
                    print(f"❌ Error loading song {filename}: {str(e)}")
        
        print(f"🎵 Total songs found: {len(songs)}")
        print("=== End Song Selection Request ===\n")
        return jsonify(songs)
    except Exception as e:
        print(f"❌ Error in get_songs: {str(e)}")
        return jsonify([])

@app.route('/get_song/<filename>')
def get_song(filename):
    try:
        print(f"🎼 Loading song: {filename}")
        song_path = os.path.join(app.root_path, 'songs', filename)
        if not os.path.exists(song_path):
            print(f"❌ Song file not found: {filename}")
            return jsonify({'error': 'Song not found'}), 404
            
        with open(song_path, 'r') as f:
            song_data = json.load(f)
            print(f"✅ Successfully loaded song: {song_data.get('title', filename)}")
            return jsonify(song_data)
    except Exception as e:
        print(f"❌ Error loading song {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/icons/<path:filename>')
def serve_icon(filename):
    try:
        icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
        return send_from_directory(icons_dir, filename)
    except Exception as e:
        print(f"Error serving icon {filename}: {e}")
        return jsonify({"error": str(e)}), 404

def generate_certificates():
    try:
        local_ip = get_local_ip()
        
        cert_path = 'certs/certificate.pem'
        key_path = 'certs/private_key.pem'
        
        if not (os.path.exists(cert_path) and os.path.exists(key_path)):
            print("\n🔒 Generating SSL certificates...")
            try:
                from OpenSSL import crypto
                
                k = crypto.PKey()
                k.generate_key(crypto.TYPE_RSA, 2048)
                
                cert = crypto.X509()
                cert.get_subject().C = "US"
                cert.get_subject().ST = "State"
                cert.get_subject().L = "City"
                cert.get_subject().O = "Organization"
                cert.get_subject().OU = "Organizational Unit"
                cert.get_subject().CN = local_ip
                cert.set_serial_number(1000)
                cert.gmtime_adj_notBefore(0)
                cert.gmtime_adj_notAfter(365*24*60*60)
                cert.set_issuer(cert.get_subject())
                cert.set_pubkey(k)
                cert.sign(k, 'sha256')
                
                os.makedirs('certs', exist_ok=True)
                
                with open(cert_path, "wb") as f:
                    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
                
                with open(key_path, "wb") as f:
                    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
                
                print("✅ SSL certificates generated successfully")
                return True
            except Exception as e:
                print(f"❌ Error generating SSL certificates: {str(e)}")
                return False
        return True
    except Exception as e:
        print(f"❌ Error in generate_certificates: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='AR Piano Learning Application')
    parser.add_argument('--port', type=int, default=8000,
                        help='Web server port')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Web server host')
    
    args = parser.parse_args()

    try:
        print("\n🎹 === AR Piano Learning Application ===")
        print("Press Ctrl+C to stop the application")
        
        local_ip = get_local_ip()
        print(f"\n🌐 Local IP address: {local_ip}")
        
        cert_path = 'certs/certificate.pem'
        key_path = 'certs/private_key.pem'
        
        if not (os.path.exists(cert_path) and os.path.exists(key_path)):
            print("\n🔒 Generating SSL certificates...")
            try:
                from OpenSSL import crypto
                
                k = crypto.PKey()
                k.generate_key(crypto.TYPE_RSA, 2048)
                
                cert = crypto.X509()
                cert.get_subject().C = "US"
                cert.get_subject().ST = "State"
                cert.get_subject().L = "City"
                cert.get_subject().O = "Organization"
                cert.get_subject().OU = "Organizational Unit"
                cert.get_subject().CN = local_ip
                cert.set_serial_number(1000)
                cert.gmtime_adj_notBefore(0)
                cert.gmtime_adj_notAfter(365*24*60*60)
                cert.set_issuer(cert.get_subject())
                cert.set_pubkey(k)
                cert.sign(k, 'sha256')
                
                os.makedirs('certs', exist_ok=True)
                
                with open(cert_path, "wb") as f:
                    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
                
                with open(key_path, "wb") as f:
                    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
                
                print("✅ SSL certificates generated successfully")
            except Exception as e:
                print(f"❌ Error generating SSL certificates: {str(e)}")
                return
        
        local_http_url = f"http://localhost:{args.port}"
        local_https_url = f"https://localhost:{args.port}"
        network_https_url = f"https://{local_ip}:{args.port}"
        
        print(f"\n🌐 Server will be available at:")
        print(f"   Desktop (HTTP): {local_http_url}")
        print(f"   Desktop (HTTPS): {local_https_url}")
        print(f"   Mobile (HTTPS): {network_https_url}")
        
        # Generate QR codes for all URLs
        http_qr = generate_qr_code(local_http_url)
        https_qr = generate_qr_code(local_https_url)
        mobile_qr = generate_qr_code(network_https_url)
        
        if http_qr and https_qr and mobile_qr:
            print_qr_code(http_qr, "Desktop")
            print_qr_code(https_qr, "Desktop")
            print_qr_code(mobile_qr, "Mobile")
        else:
            print("\n❌ Could not generate QR codes")
            return
        
        print("\n📋 Instructions:")
        print("1. Open this URL on your desktop: " + local_http_url)
        print("2. Scan the QR code with your phone")
        print("3. On your phone:")
        print("   - Accept the security warning")
        print("   - Click 'Advanced' if needed")
        print("   - Click 'Proceed to site'")
        print("4. Click 'Start Camera' on your phone")
        print("5. The video feed will appear on your desktop")
        
        midi_connected = recognition.initialize_midi()
        print(f"\n🎹 MIDI keyboard: {'✅ Connected' if midi_connected else '❌ Not connected (running in visual-only mode)'}")
        
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((args.host, args.port))
            sock.close()
            print(f"\n✅ Port {args.port} is available")
        except socket.error as e:
            print(f"\n❌ Port {args.port} is already in use or blocked")
            print(f"Error: {str(e)}")
            print("\n🔧 Please try:")
            print("1. Close any other applications using port", args.port)
            print("2. Try a different port (e.g., --port 8001)")
            return

        print("\n🚀 Starting server...")
        
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        context.verify_mode = ssl.CERT_NONE
        context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3
        context.options |= ssl.OP_SINGLE_DH_USE | ssl.OP_SINGLE_ECDH_USE
        context.set_ciphers('ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384')
        
        socketio.run(app, 
                    host=args.host, 
                    port=args.port,
                    ssl_context=context,
                    allow_unsafe_werkzeug=True)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure no other application is using port", args.port)
        print("2. Check your firewall settings")
        print("3. Try running as administrator")
        print("4. Try using a different port (e.g., --port 8001)")
        
    finally:
        if recognition.midi_input:
            recognition.midi_input.close()
        if recognition.camera:
            recognition.camera.release()

@socketio.on('start_learning_mode')
def handle_start_learning():
    recognition.start_piano_learning_mode()

@socketio.on('stop_learning_mode')
def handle_stop_learning():
    recognition.stop_piano_learning_mode()

@socketio.on('request_next_note')
def handle_next_note():
    if recognition.piano_learning_mode:
        next_note = recognition.get_next_note()
        socketio.emit('next_note', {'note': next_note})

@socketio.on('playmode_debug')
def handle_playmode_debug(data):
    print('\n[PlayMode Debug]', json.dumps(data, indent=2))

if __name__ == '__main__':
    main()
