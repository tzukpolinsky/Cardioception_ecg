from pydub import AudioSegment
from pydub.playback import play
import time
import sys

def repeat_tone_for_duration(file_path, total_duration_sec, bpm):
    # Load the tone
    try:
        tone = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Turns volume up
    gain_db = 6 # Max value is 9
    if gain_db != 0:
        tone = tone.apply_gain(gain_db)

    # Calculate beat interval (seconds per beat)
    beat_interval = 60 / bpm
    tone_duration = tone.duration_seconds

    if tone_duration > beat_interval:
        print("Warning: Tone is longer than beat interval. Overlapping will occur.")

    # Compute how many beats fit in total_duration_sec
    num_beats = int(total_duration_sec / beat_interval)

    print(f"Playing for {total_duration_sec} seconds at {bpm} BPM ({num_beats} beats)")

    start_time = time.time()
    for _ in range(num_beats):
        play(tone)
        elapsed = time.time() - start_time
        if elapsed >= total_duration_sec:
            break
        time.sleep(max(0, beat_interval - tone_duration))

    return num_beats
# === Example usage ===
if __name__ == "__main__":
    repeat_tone_for_duration("one_heartbeat.wav", total_duration_sec=15, bpm=60)
