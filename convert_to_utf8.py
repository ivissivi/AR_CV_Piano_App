import os
import json
import codecs

def convert_file_to_utf8(file_path):
    print(f"\nProcessing {file_path}")
    
    # First, try to detect the current encoding
    with open(file_path, 'rb') as f:
        raw_content = f.read()
        print(f"Raw content starts with: {raw_content[:20]}")
    
    # Try to decode with different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'cp850']
    decoded_content = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            decoded_content = raw_content.decode(encoding)
            used_encoding = encoding
            print(f"Successfully decoded with {encoding}")
            break
        except UnicodeDecodeError as e:
            print(f"Failed to decode with {encoding}: {str(e)}")
            continue
    
    if decoded_content is None:
        print(f"Could not decode {file_path} with any encoding")
        return
    
    # Parse the JSON to validate it
    try:
        data = json.loads(decoded_content)
        print(f"Successfully parsed JSON")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {file_path}: {e}")
        return
    
    # Write back with UTF-8 encoding
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully wrote file with UTF-8 encoding")
        
        # Verify the file was written correctly
        with open(file_path, 'r', encoding='utf-8') as f:
            verification = f.read()
            print(f"Verification read successful")
    except Exception as e:
        print(f"Error writing file: {str(e)}")

def main():
    songs_dir = os.path.join(os.path.dirname(__file__), 'songs')
    print(f"Converting files in directory: {songs_dir}")
    
    for filename in os.listdir(songs_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(songs_dir, filename)
            convert_file_to_utf8(file_path)

if __name__ == '__main__':
    main() 