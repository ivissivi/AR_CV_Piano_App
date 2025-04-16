# AR_CV_Piano_App

An augmented reality application that helps you learn to play the piano by providing visual feedback and real-time note recognition.

## Features

- Real-time piano key detection using computer vision
- MIDI keyboard support for interactive learning
- Visual guidance for playing songs
- Note sound playback
- Web interface for easy access

## Requirements

- Python 3.7+
- OpenCV (cv2)
- Flask
- Mido (for MIDI support)
- Pygame (for sound playback)
- A camera (webcam or IP camera)
- MIDI keyboard (optional)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ivissivi/AR_CV_Piano_App.git
cd AR_CV_Piano_App
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure you have the following files in place:
- `piano_key_mapping.json` (included)
- `notes/` directory with WAV files for each note (C.wav, C#.wav, etc.)

## Usage

Run the application with default settings:
```bash
python ar_cv.py
```

### Command Line Options

- `--camera`: Camera source (default: 0)
  - Webcam index (e.g., 0, 1)
  - IP camera URL (e.g., http://192.168.1.100:8080/video)
  - Video file path
- `--port`: Web server port (default: 5000)
- `--host`: Web server host (default: 0.0.0.0)

Example with custom settings:
```bash
python ar_cv.py --camera 1 --port 8080
```

### Using the Application

1. Open your web browser and navigate to `http://localhost:5000`
2. Position your camera to view the piano keys
3. The application will detect and outline the piano keys
4. If you have a MIDI keyboard connected, you can play along with the visual guide
5. The red line indicates the next note to play in the song

## Troubleshooting

- If the camera isn't detected, try a different camera source
- If MIDI keyboard isn't working, check your connections and try reconnecting
- Make sure all note sound files are present in the `notes/` directory
- Ensure proper lighting for key detection

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
 
