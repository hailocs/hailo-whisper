# numpy implementation of audio_utils.py from Whisper repo
import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
from scipy import signal


def exact_div(x, y):
    assert x % y == 0
    return x // y


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES
    """
    if array.shape[axis] > length:
        array = np.take(array, range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> np.ndarray:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.    Allows decoupling librosa dependency; saved using:
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return f[f"mel_{n_mels}"]


def log_mel_spectrogram(
    audio: Union[str, np.ndarray],
    n_mels: int = 80,
    padding: int = 0,
):
    """
    Compute the log-Mel spectrogram

    Parameters
    ----------
    audio: Union[str, np.ndarray], shape = (*)
        The path to audio or a NumPy array containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    Returns
    -------
    np.ndarray, shape = (n_mels, n_frames)
        An array that contains the Mel spectrogram
    """
    if isinstance(audio, str):
        audio = load_audio(audio)

    if padding > 0:
        audio = np.pad(audio, (0, padding))

    # Create Hann window
    window = signal.windows.hann(N_FFT, sym=False)

    # Compute STFT
    frequencies, times, stft = signal.stft(
        audio,
        fs=SAMPLE_RATE,
        window=window,
        nperseg=N_FFT,
        noverlap=N_FFT-HOP_LENGTH,
        boundary=None,
        padded=False
    )

    # Compute magnitude spectrogram (excluding last frame as in PyTorch version)
    magnitudes = np.abs(stft[..., :-1]) ** 2

    # Apply mel filterbank
    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    # Convert to log scale with the same scaling as the PyTorch version
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec