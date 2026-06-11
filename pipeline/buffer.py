import numpy as np
from collections import deque


class AudioBuffer:
    def __init__(self, maxlen: int = 10):
        self.buffer = deque(maxlen=maxlen)

    def push(self, chunk: np.ndarray):
        self.buffer.append(chunk.copy())

    def pop_all(self) -> np.ndarray:
        if not self.buffer:
            return np.array([], dtype=np.float32)
        result = np.concatenate(list(self.buffer))
        self.buffer.clear()
        return result

    def peek_all(self) -> np.ndarray:
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self.buffer))

    @property
    def size(self):
        return sum(len(c) for c in self.buffer)

    def clear(self):
        self.buffer.clear()


class CrossFader:
    def __init__(self, fade_len: int = 128):
        self.fade_len = fade_len
        self.fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        self.fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)

    def crossfade(self, prev: np.ndarray, next_chunk: np.ndarray) -> np.ndarray:
        if len(prev) < self.fade_len or len(next_chunk) < self.fade_len:
            return next_chunk
        overlap = np.concatenate([
            prev[-self.fade_len:] * self.fade_out,
            next_chunk[:self.fade_len] * self.fade_in,
        ])
        return np.concatenate([
            next_chunk[:self.fade_len],
            overlap,
            next_chunk[self.fade_len:],
        ])
