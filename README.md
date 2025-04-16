# AR CV Piano App

An interactive piano application that uses computer vision and hand tracking to create an immersive augmented reality piano experience. The app allows users to play a virtual piano using hand gestures captured through their device's camera, providing a unique and engaging way to interact with music.

## Features

- **Hand Tracking**: Uses MediaPipe for real-time hand tracking and gesture recognition
- **Dual Mode Support**:
  - **Mobile Mode**: View and play directly on your mobile device
  - **Desktop Mode**: Stream the piano interface to a desktop computer
- **Camera Support**:
  - Optimized for both front and back cameras
  - Automatic back camera selection on iOS devices
  - Smooth camera switching
- **User Interface**:
  - Clean, modern design
  - Auto-hiding status messages
  - Subtle hand tracking status indicator
  - Fullscreen mode support
- **Performance Optimizations**:
  - Efficient hand tracking processing
  - Frame rate optimization
  - Responsive UI updates

## Requirements

- Python 3.7+
- Flask
- OpenCV
- MediaPipe
- Socket.IO
- A modern web browser with WebRTC support
- A device with a camera (mobile device recommended)

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

3. Choose your viewing mode:
   - **View on Phone**: Use your mobile device's camera directly
   - **Stream to Desktop**: Stream the piano interface to a desktop computer

4. Camera Controls:
   - The app will automatically select the back camera on iOS devices
   - Use the camera controls to start/stop the camera or toggle fullscreen mode

5. Hand Tracking:
   - Position your hands in view of the camera
   - The app will track your hand movements and translate them into piano notes
   - The hand tracking status indicator shows when hands are detected

## Technical Details

- **Hand Tracking**: Uses MediaPipe's hand tracking solution with optimized settings for performance
- **Camera Selection**: Implements intelligent camera selection based on device type and capabilities
- **UI/UX**: Responsive design with smooth transitions and intuitive controls
- **Performance**: Optimized frame processing and hand tracking for smooth operation

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
 
