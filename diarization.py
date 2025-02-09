from pyannote.audio import Pipeline


def diarize_speakers(audio_file, pretrained_model='pyannote/speaker-diarization-3.1'):
    """
    Runs speaker diarization on an audio file.
    Returns a list of (start_time, end_time, speaker_label).
    """
    # Load pre-trained diarization pipeline
    pipeline = Pipeline.from_pretrained(pretrained_model,
                                        use_auth_token="")

    # Perform diarization
    diarization = pipeline(audio_file)

    # Collate results into a list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        segments.append((start_time, end_time, speaker))
    return segments


def write_srt_diarized(segments, output_srt):
    """
    Writes a minimal SRT with speaker labels (optional) and blank lines for text.
    segments: list of (start_time, end_time, speaker_label).
    """

    def srt_time_format(seconds):
        import math
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    # Sort segments by start time to ensure correct SRT order
    segments = sorted(segments, key=lambda x: x[0])

    with open(output_srt, 'w', encoding='utf-8') as f:
        for i, (start, end, speaker) in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{srt_time_format(start)} --> {srt_time_format(end)}\n")
            # Put speaker label as a placeholder comment or prefix
            f.write(f"[{speaker}]: \n\n")


if __name__ == "__main__":
    AUDIO_FILE = "example_audio.wav"
    OUTPUT_SRT = "diarization_output.srt"

    diarized_segments = diarize_speakers(AUDIO_FILE)
    write_srt_diarized(diarized_segments, OUTPUT_SRT)
    print(f"Generated {OUTPUT_SRT} with {len(diarized_segments)} segments.")
