import React, { useState, useEffect } from 'react';

const SongSelection = () => {
    const [songs, setSongs] = useState([]);
    const [selectedSong, setSelectedSong] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadSongs = async () => {
            try {
                const response = await fetch('/songs');
                if (!response.ok) {
                    throw new Error('Failed to load songs');
                }
                const data = await response.json();
                setSongs(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        loadSongs();
    }, []);

    const handleSongSelect = (song) => {
        setSelectedSong(song);
    };

    if (loading) return <div className="text-center p-4">Loading songs...</div>;
    if (error) return <div className="text-center p-4 text-red-500">Error: {error}</div>;

    return (
        <div className="p-4">
            <h2 className="text-2xl font-bold mb-4">Select a Song</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {songs.map((song) => (
                    <div
                        key={song.title}
                        className={`p-4 rounded-lg cursor-pointer transition-all ${
                            selectedSong?.title === song.title
                                ? 'bg-blue-100 border-2 border-blue-500'
                                : 'bg-white border border-gray-200 hover:border-blue-300'
                        }`}
                        onClick={() => handleSongSelect(song)}
                    >
                        <h3 className="text-lg font-semibold mb-2">{song.title}</h3>
                        <div className="space-y-1 text-sm text-gray-600">
                            <p><span className="font-medium">Difficulty:</span> {song.difficulty}</p>
                            <p><span className="font-medium">Tempo:</span> {song.tempo} BPM</p>
                            <p><span className="font-medium">Time Signature:</span> {song.time_signature}</p>
                            <p><span className="font-medium">Key:</span> {song.key}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default SongSelection; 