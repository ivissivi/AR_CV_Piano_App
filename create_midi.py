from midiutil import MIDIFile
import os

def create_mary_had_a_little_lamb():
    # Create a MIDI file with one track
    midi = MIDIFile(1)
    
    # Add track name and tempo
    track = 0
    time = 0
    midi.addTrackName(track, time, "Mary Had a Little Lamb")
    midi.addTempo(track, time, 120)  # 120 BPM
    
    # Notes for Mary Had a Little Lamb (in MIDI note numbers)
    # E4=64, D4=62, C4=60, G4=67
    notes = [
        (64, 0.5),    # E
        (62, 0.5),    # D
        (60, 0.5),    # C
        (62, 0.5),    # D
        (64, 0.5),    # E
        (64, 0.5),    # E
        (64, 0.5),    # E
        (None, 0.5),  # rest
        (62, 0.5),    # D
        (62, 0.5),    # D
        (62, 0.5),    # D
        (None, 0.5),  # rest
        (64, 0.5),    # E
        (67, 0.5),    # G
        (67, 0.5),    # G
        (None, 0.5),  # rest
        (64, 0.5),    # E
        (62, 0.5),    # D
        (60, 0.5),    # C
        (62, 0.5),    # D
        (64, 0.5),    # E
        (64, 0.5),    # E
        (64, 0.5),    # E
        (64, 0.5),    # E
        (62, 0.5),    # D
        (62, 0.5),    # D
        (64, 0.5),    # E
        (62, 0.5),    # D
        (60, 0.5),    # C
    ]
    
    # Add notes to the MIDI file
    for note, duration in notes:
        if note is not None:
            midi.addNote(track, 0, note, time, duration, 100)  # 100 is velocity (volume)
        time += duration
    
    # Create songs directory if it doesn't exist
    if not os.path.exists('songs'):
        os.makedirs('songs')
    
    # Write the MIDI file
    with open('songs/mary_had_a_little_lamb.mid', 'wb') as f:
        midi.writeFile(f)

if __name__ == '__main__':
    create_mary_had_a_little_lamb() 