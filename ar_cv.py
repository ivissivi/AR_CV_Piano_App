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
import time
import sys
import ssl
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
DEFAULT_PORT = 5000
DEFAULT_HOST = '0.0.0.0'

# Set MIDI backend to use rtmidi
mido.set_backend('mido.backends.rtmidi')

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
    try:
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
    except Exception as e:
        print(f"\n❌ Error generating QR code: {str(e)}")
        return None

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
        self.midi_thread = None
        self.midi_running = False
        self.camera = None
        self.camera_error = None

    def initialize_camera(self):
        """Initialize the camera with proper error handling"""
        try:
            print("\n📸 Initializing camera...")
            if self.camera is not None:
                self.camera.release()
            
            # Try to open the camera with different backends
            for i in range(3):  # Try up to 3 different camera indices
                self.camera = cv2.VideoCapture(i)
                if self.camera.isOpened():
                    print(f"✅ Camera {i} opened successfully")
                    break
                else:
                    self.camera.release()
            
            if not self.camera.isOpened():
                raise Exception("No camera could be opened")
            
            # Set camera properties for better mobile compatibility
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test camera by reading a frame
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
        """Run in a separate thread to handle MIDI input"""
        try:
            print("\n🎹 === Starting MIDI Listener Thread ===")
            mido.set_backend('mido.backends.rtmidi')
            time.sleep(1)  # Give some time for other resources to initialize
            
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
                        self.midi_port = port  # Store the port reference
                        self.midi_running = True
                        
                        print("\n🎵 Ready to receive MIDI messages!")
                        print("   Press keys on your MIDI keyboard to test")
                        print("   Press Ctrl+C to stop")
                        
                        while self.midi_running:
                            for msg in port.iter_pending():
                                if msg.type == 'note_on' and msg.velocity > 0:
                                    note = msg.note
                                    print(f"\n🎼 Note pressed: {note} (Velocity: {msg.velocity})")
                                    # Process the note here
                except Exception as e:
                    print(f"\n❌ Error in MIDI thread: {str(e)}")
            else:
                print(f"\n❌ Could not find {target_port} in available ports")
        except Exception as e:
            print(f"\n❌ MIDI thread error: {str(e)}")
        finally:
            self.midi_running = False
            self.midi_port = None

    def initialize_midi(self):
        """Start the MIDI listener thread"""
        try:
            if self.midi_thread is None or not self.midi_thread.is_alive():
                self.midi_running = True
                self.midi_thread = threading.Thread(target=self.midi_listener)
                self.midi_thread.daemon = True  # Thread will exit when main program exits
                self.midi_thread.start()
                print("\n🔄 Starting MIDI listener thread...")
                return True
            return False
        except Exception as e:
            print(f"\n❌ Error starting MIDI thread: {str(e)}")
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
        """Generate video frames with proper error handling"""
        if not self.initialize_camera():
            # Generate error frame
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

                # Resize frame for better performance
                frame = cv2.resize(frame, (640, 480))
                
                # Process the frame
                processed_frame = self.detect_and_outline_keys(frame)
                
                # Encode the frame
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"\n❌ Frame generation error: {str(e)}")
                break

    def __del__(self):
        """Cleanup when the object is destroyed"""
        print("\n🛑 Cleaning up resources...")
        self.midi_running = False
        if self.midi_thread and self.midi_thread.is_alive():
            self.midi_thread.join(timeout=1.0)
        if self.midi_port:
            try:
                self.midi_port.close()
                print("✅ MIDI port closed successfully")
            except:
                print("❌ Error closing MIDI port")
        if self.camera:
            try:
                self.camera.release()
                print("✅ Camera released successfully")
            except:
                print("❌ Error releasing camera")

# Initialize recognition system
recognition = PianoKeyRecognition()

# Video stream class to manage frames and clients
class VideoStream:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 15  # 15 FPS target

    def update(self, frame):
        current_time = time.time()
        with self.lock:
            # Only update if enough time has passed
            if current_time - self.last_frame_time >= self.frame_interval:
                self.frame = frame
                self.last_frame_time = current_time

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

video_stream = VideoStream()

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    """Handle incoming video frames from mobile clients"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(data)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
            
        # Resize frame if too large
        height, width = frame.shape[:2]
        if width > 640 or height > 480:
            frame = cv2.resize(frame, (640, 480))
            
        # Update video stream
        video_stream.update(frame)
        
        # Re-encode frame with lower quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            return
            
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Broadcast frame to all connected desktop clients
        emit('frame', frame_data, broadcast=True)
    except Exception as e:
        print(f"Error processing frame: {str(e)}")

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

def generate_certificates():
    """Generate SSL certificates if they don't exist"""
    try:
        # Get local IP address
        local_ip = get_local_ip()
        
        # Check if certificates exist
        cert_path = 'certs/certificate.pem'
        key_path = 'certs/private_key.pem'
        
        if not (os.path.exists(cert_path) and os.path.exists(key_path)):
            print("\n🔒 Generating SSL certificates...")
            try:
                from OpenSSL import crypto
                
                # Create a key pair
                k = crypto.PKey()
                k.generate_key(crypto.TYPE_RSA, 2048)
                
                # Create a self-signed cert
                cert = crypto.X509()
                cert.get_subject().C = "US"
                cert.get_subject().ST = "State"
                cert.get_subject().L = "City"
                cert.get_subject().O = "Organization"
                cert.get_subject().OU = "Organizational Unit"
                cert.get_subject().CN = local_ip
                cert.set_serial_number(1000)
                cert.gmtime_adj_notBefore(0)
                cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for 1 year
                cert.set_issuer(cert.get_subject())
                cert.set_pubkey(k)
                cert.sign(k, 'sha256')
                
                # Create certs directory if it doesn't exist
                os.makedirs('certs', exist_ok=True)
                
                # Write certificate
                with open(cert_path, "wb") as f:
                    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
                
                # Write private key
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
        
        # Get local IP address
        local_ip = get_local_ip()
        print(f"\n🌐 Local IP address: {local_ip}")
        
        # Check if certificates exist
        cert_path = 'certs/certificate.pem'
        key_path = 'certs/private_key.pem'
        
        if not (os.path.exists(cert_path) and os.path.exists(key_path)):
            print("\n🔒 Generating SSL certificates...")
            try:
                # Generate certificates if they don't exist
                from OpenSSL import crypto
                
                # Create a key pair
                k = crypto.PKey()
                k.generate_key(crypto.TYPE_RSA, 2048)
                
                # Create a self-signed cert
                cert = crypto.X509()
                cert.get_subject().C = "US"
                cert.get_subject().ST = "State"
                cert.get_subject().L = "City"
                cert.get_subject().O = "Organization"
                cert.get_subject().OU = "Organizational Unit"
                cert.get_subject().CN = local_ip
                cert.set_serial_number(1000)
                cert.gmtime_adj_notBefore(0)
                cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for 1 year
                cert.set_issuer(cert.get_subject())
                cert.set_pubkey(k)
                cert.sign(k, 'sha256')
                
                # Create certs directory if it doesn't exist
                os.makedirs('certs', exist_ok=True)
                
                # Write certificate
                with open(cert_path, "wb") as f:
                    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
                
                # Write private key
                with open(key_path, "wb") as f:
                    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
                
                print("✅ SSL certificates generated successfully")
            except Exception as e:
                print(f"❌ Error generating SSL certificates: {str(e)}")
                return
        
        # Generate URLs
        local_http_url = f"http://localhost:{args.port}"
        local_https_url = f"https://localhost:{args.port}"
        network_https_url = f"https://{local_ip}:{args.port}"
        
        print(f"\n🌐 Server will be available at:")
        print(f"   Desktop (HTTP): {local_http_url}")
        print(f"   Desktop (HTTPS): {local_https_url}")
        print(f"   Mobile (HTTPS): {network_https_url}")
        
        # Generate QR code with network URL
        recognition.qr_code = generate_qr_code(network_https_url)
        if recognition.qr_code is None:
            print("\n❌ Could not generate QR code")
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
        
        # Check if port is available
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

        # Run the server with both HTTP and HTTPS
        print("\n🚀 Starting server...")
        
        # Create SSL context with proper settings
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        context.verify_mode = ssl.CERT_NONE
        context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3
        context.options |= ssl.OP_SINGLE_DH_USE | ssl.OP_SINGLE_ECDH_USE
        context.set_ciphers('ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384')
        
        # Run the server with WebSocket support
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
        if recognition.midi_port:
            recognition.midi_port.close()
        if recognition.camera:
            recognition.camera.release()

if __name__ == '__main__':
    main()
