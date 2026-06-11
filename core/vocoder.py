import numpy as np
import librosa


class Vocoder:
    def __init__(self, n_fft=1024, hop_length=320, win_length=1024, n_mels=80, sr=24000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sr = sr

        self.mel_basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=sr // 2
        )
        self.mel_basis_inv = np.linalg.pinv(self.mel_basis)

    def decode(self, mel: np.ndarray) -> np.ndarray:
        if mel.ndim == 3:
            mel = mel.squeeze(0)

        if mel.ndim != 2:
            return np.zeros(self.hop_length * mel.shape[-1])

        mel_db = mel * 80 - 80
        mel_linear = librosa.db_to_power(mel_db)

        linear_spec = self.mel_basis_inv @ mel_linear
        linear_spec = np.maximum(linear_spec, 1e-10)

        audio = librosa.griffinlim(
            linear_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_iter=16,
        )
        audio = np.nan_to_num(audio, nan=0.0)
        return audio.astype(np.float32)
