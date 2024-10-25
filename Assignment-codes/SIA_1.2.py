import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Record audio
duration = 5
sample_rate = 44100
print("Recording audio...")
audio_signal = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("Recording complete.")

# Flatten the audio signal
audio_signal = audio_signal.flatten()

# Create delayed signal
delay_seconds = 0.1
delay_samples = int(delay_seconds * sample_rate)
delayed_signal = np.roll(audio_signal, delay_samples)

# Compute cross-correlation
cross_corr = np.correlate(audio_signal, delayed_signal, mode='full')

# Create time arrays
time = np.arange(len(audio_signal)) / sample_rate
lag = np.arange(-len(audio_signal) + 1, len(audio_signal)) / sample_rate

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
ax1.plot(time, audio_signal)
ax1.set_title("Original Signal")
ax2.plot(time, delayed_signal)
ax2.set_title(f"Delayed Signal (Delay: {delay_seconds} seconds)")
ax3.plot(lag, cross_corr)
ax3.set_title("Cross-correlation")
ax3.set_xlabel("Lag (seconds)")
plt.tight_layout()
plt.show()