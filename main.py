import time

import mlx.core as mx
import numpy as np
from gigaam_mlx import load_model
from gigaam_mlx.audio import compute_mel, load_audio, split_audio

from converter import extract_audio

total_start = time.perf_counter()

video_path = "lecture.mp4"
audio_path = "lecture_audio.mp3"

lecture = extract_audio(video_path, audio_path)

model, tokenizer = load_model(model_type="rnnt")

audio = load_audio(audio_path)
chunks = split_audio(audio, max_chunk_sec=20)

texts = []

for i, chunk in enumerate(chunks):
    start = chunk["start_sample"]
    end = chunk["end_sample"]

    chunk_audio = audio[start:end]

    mel = compute_mel(chunk_audio)
    mel_mx = mx.array(mel[np.newaxis])

    encoded, seq_len = model.encode(mel_mx)
    mx.eval(encoded)

    token_ids = model.decode(encoded, seq_len)
    text = tokenizer.decode(token_ids)

    texts.append(text)

    print(f"{i + 1}/{len(chunks)}")

text = " ".join(texts)

with open("lecture_transcript.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Total time: {time.perf_counter() - total_start:.3f} seconds")