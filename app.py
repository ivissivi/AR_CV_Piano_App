import os
import json
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/get_songs')
def get_songs():
    songs_dir = os.path.join(os.path.dirname(__file__), 'songs')
    songs = []
    
    # Get all JSON files in the songs directory
    for filename in os.listdir(songs_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(songs_dir, filename), 'r') as f:
                    song_data = json.load(f)
                    songs.append({
                        'title': song_data.get('title', filename),
                        'filename': filename,
                        'tempo': song_data.get('tempo', 0),
                        'difficulty': song_data.get('difficulty', 'unknown'),
                        'time_signature': song_data.get('time_signature', '4/4'),
                        'key': song_data.get('key', 'C major')
                    })
            except Exception as e:
                print(f"Error loading song {filename}: {e}")
    
    return jsonify(songs)

@app.route('/get_song/<path:filename>')
def get_song(filename):
    try:
        songs_dir = os.path.join(os.path.dirname(__file__), 'songs')
        file_path = os.path.join(songs_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "Song not found"}), 404
            
        with open(file_path, 'r') as f:
            song_data = json.load(f)
            return jsonify(song_data)
    except Exception as e:
        print(f"Error serving song {filename}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/songs/<path:filename>')
def serve_song(filename):
    try:
        songs_dir = os.path.join(os.path.dirname(__file__), 'songs')
        return send_from_directory(songs_dir, filename)
    except Exception as e:
        print(f"Error serving song {filename}: {e}")
        return jsonify({"error": str(e)}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 