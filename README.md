# AR CV Piano App

An interactive piano application that uses computer vision and hand tracking to create an immersive augmented reality piano experience. The app allows users to play a virtual piano using hand gestures captured through their device's camera, providing a unique and engaging way to interact with music through modern technology.

## Features

- **Hand Tracking**: 
  - Real-time hand tracking and gesture recognition using MediaPipe
  - Optimized performance for smooth interaction
  - Accurate finger position detection
  - Dynamic gesture mapping to piano keys
- **Dual Mode Support**:
  - **Mobile Mode**: View and play directly on your mobile device
  - **Desktop Mode**: Stream the piano interface to a desktop computer
  - Seamless mode switching
- **Camera Support**:
  - Optimized for both front and back cameras
  - Automatic back camera selection on iOS devices
  - Smooth camera switching with UI controls
  - Camera selection interface for multiple cameras
- **User Interface**:
  - Clean, modern design
  - Auto-hiding status messages
  - Subtle hand tracking status indicator
  - Fullscreen mode support
  - Responsive layout for all devices
  - Camera controls with visual feedback
- **Performance Optimizations**:
  - Efficient hand tracking processing
  - Frame rate optimization
  - Responsive UI updates
  - Optimized video streaming
  - Reduced latency for real-time interaction

## Requirements

- Python 3.7+
- Flask
- OpenCV
- MediaPipe
- Socket.IO
- A modern web browser with WebRTC support
- A device with a camera (mobile device recommended)
- SSL certificates for secure connection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ivissivi/AR_CV_Piano_App.git
cd AR_CV_Piano_App
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Generate SSL certificates (for secure WebRTC connections):
```bash
python generate_cert.py
```

## Usage

1. Start the server:
```bash
python ar_cv.py
```

2. Open the application in your browser:
   - For mobile: Use the provided IP address and port
   - For desktop: Use the same IP address and port
   - Make sure to use HTTPS for secure connection

3. Choose your viewing mode:
   - **View on Phone**: Use your mobile device's camera directly
   - **Stream to Desktop**: Stream the piano interface to a desktop computer

4. Camera Controls:
   - The app will automatically select the back camera on iOS devices
   - Use the camera controls to:
     - Start/stop the camera
     - Switch between front and back cameras
     - Toggle fullscreen mode
     - Select specific camera (if multiple available)

5. Hand Tracking:
   - Position your hands in view of the camera
   - The app will track your hand movements and translate them into piano notes
   - The hand tracking status indicator shows when hands are detected
   - Adjust your hand position based on the visual feedback

## Technical Details

- **Hand Tracking**: 
  - Uses MediaPipe's hand tracking solution with optimized settings
  - Custom gesture recognition algorithms
  - Real-time finger position tracking
- **Camera Selection**: 
  - Implements intelligent camera selection based on device type
  - Supports multiple camera configurations
  - Automatic iOS back camera detection
- **UI/UX**: 
  - Responsive design with smooth transitions
  - Intuitive camera controls
  - Real-time visual feedback
  - Auto-hiding interface elements
- **Performance**: 
  - Optimized frame processing
  - Efficient hand tracking algorithms
  - Reduced latency video streaming
  - Smooth UI transitions

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
 
