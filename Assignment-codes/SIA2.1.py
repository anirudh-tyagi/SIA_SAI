import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import signal

def record_audio(duration=3, fs=44100):
    """Record audio from microphone"""
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording finished")
    return audio.flatten()

def sample_signal(original_signal, original_fs, target_fs):
    """Sample the signal at given frequency"""
    # Calculate the sampling ratio
    ratio = int(original_fs / target_fs)
    # Take every nth sample
    sampled = original_signal[::ratio]
    return sampled

def reconstruct_signal(sampled_signal, original_length):
    """Reconstruct signal using linear interpolation"""
    # Create time points for interpolation
    x = np.linspace(0, len(sampled_signal), len(sampled_signal))
    x_new = np.linspace(0, len(sampled_signal), original_length)
    # Interpolate
    reconstructed = np.interp(x_new, x, sampled_signal)
    return reconstructed

def main():
    # Parameters
    duration = 3  # seconds
    original_fs = 44100  # Original sampling rate
    nyquist_fs = 8000    # Nyquist rate (typical for speech)
    under_fs = 4000      # Under-sampling rate
    over_fs = 16000      # Over-sampling rate

    # Record audio
    original_signal = record_audio(duration, original_fs)

    # time axis for plotting
    t = np.linspace(0, duration, len(original_signal))

    # Sample at different rates
    nyquist_sampled = sample_signal(original_signal, original_fs, nyquist_fs)
    under_sampled = sample_signal(original_signal, original_fs, under_fs)
    over_sampled = sample_signal(original_signal, original_fs, over_fs)

    # Reconstruct signals
    nyquist_reconstructed = reconstruct_signal(nyquist_sampled, len(original_signal))
    under_reconstructed = reconstruct_signal(under_sampled, len(original_signal))
    over_reconstructed = reconstruct_signal(over_sampled, len(original_signal))

    # Plotting
    plt.figure(figsize=(15, 10))

    # Original signal
    plt.subplot(4, 1, 1)
    plt.plot(t, original_signal)
    plt.title('Original Signal')
    plt.ylabel('Amplitude')

    # Nyquist rate sampling
    plt.subplot(4, 1, 2)
    plt.plot(t, nyquist_reconstructed)
    plt.title(f'Nyquist Rate Sampling ({nyquist_fs} Hz) and Reconstruction')
    plt.ylabel('Amplitude')

    # Under-sampling
    plt.subplot(4, 1, 3)
    plt.plot(t, under_reconstructed)
    plt.title(f'Under-sampling ({under_fs} Hz) and Reconstruction')
    plt.ylabel('Amplitude')

    # Over-sampling
    plt.subplot(4, 1, 4)
    plt.plot(t, over_reconstructed)
    plt.title(f'Over-sampling ({over_fs} Hz) and Reconstruction')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # Playing original and reconstructed signals
    print("\nPlaying original signal...")
    sd.play(original_signal, original_fs)
    sd.wait()

    print("Playing reconstructed signal (Nyquist rate)...")
    sd.play(nyquist_reconstructed, original_fs)
    sd.wait()

if __name__ == "__main__":
    main()