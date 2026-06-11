import json
import os
import subprocess
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf


def _ensure_ffmpeg():
    paths = [
        r"C:\Users\Developer\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin",
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
    ]
    for p in paths:
        if os.path.isdir(p):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            return
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
    except FileNotFoundError:
        print("  FFmpeg no encontrado. Instalalo o convertí el audio a WAV.")


class AudioPreprocessor:
    def __init__(self, config_path="config/settings.json"):
        _ensure_ffmpeg()
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.sr = self.cfg["training"]["sr"]

    def run(self, input_path: str, output_dir: str):
        audio_path = Path(input_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Preprocesando: {audio_path}")
        audio, orig_sr = librosa.load(str(audio_path), sr=None, mono=True)

        if orig_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sr)

        # Limit to first 90 seconds for faster training
        max_samples = 90 * self.sr
        if len(audio) > max_samples:
            print(f"  Audio largo ({len(audio)/self.sr:.0f}s), truncando a 90s")
            audio = audio[:max_samples]

        segments = self._segment_audio(audio)

        for i, seg in enumerate(segments):
            seg_path = output_path / f"segment_{i:04d}.wav"
            sf.write(str(seg_path), seg, self.sr)

        # Generate training metadata
        metadata = self._generate_metadata(segments)
        meta_path = output_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        total_duration = sum(len(s) for s in segments) / self.sr
        print(f"  Segmentos: {len(segments)}")
        print(f"  Duración total: {total_duration:.1f}s")
        print(f"  Output: {output_path}")

        return str(output_path)

    def _segment_audio(self, audio: np.ndarray) -> list:
        from audio import VoiceActivityDetector

        vad = VoiceActivityDetector()
        segments = []

        frame_len = int(0.03 * self.sr)
        hop_len = frame_len // 2
        min_segment = int(1.0 * self.sr)
        max_segment = int(15.0 * self.sr)

        speech_regions = []
        in_speech = False
        start = 0

        for pos in range(0, len(audio) - frame_len, hop_len):
            frame = audio[pos : pos + frame_len]
            is_speech = vad.process_frame(frame, self.sr)

            if is_speech and not in_speech:
                start = pos
                in_speech = True
            elif not is_speech and in_speech:
                end = pos + frame_len
                if end - start >= min_segment:
                    speech_regions.append((start, min(end, len(audio))))
                in_speech = False

        if in_speech:
            speech_regions.append((start, len(audio)))

        for start, end in speech_regions:
            seg = audio[start:end]
            if len(seg) > max_segment:
                for sub_start in range(start, end, max_segment):
                    sub_end = min(sub_start + max_segment, end)
                    segments.append(audio[sub_start:sub_end])
            else:
                segments.append(seg)

        if not segments:
            segments.append(audio)

        return segments

    def _generate_metadata(self, segments: list) -> dict:
        total_samples = sum(len(s) for s in segments)
        return {
            "num_segments": len(segments),
            "total_samples": int(total_samples),
            "sample_rate": self.sr,
            "total_duration_seconds": float(total_samples / self.sr),
            "segments": [
                {
                    "index": i,
                    "samples": len(seg),
                    "duration": len(seg) / self.sr,
                }
                for i, seg in enumerate(segments)
            ],
        }
