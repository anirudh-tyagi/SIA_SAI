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

# Compute autocorrelation
autocorr = np.correlate(audio_signal, audio_signal, mode='full')

# Create time arrays
time = np.arange(len(audio_signal)) / sample_rate
lag = np.arange(-len(audio_signal) + 1, len(audio_signal)) / sample_rate

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(time, audio_signal)
ax1.set_title("Recorded Audio Signal")
ax1.set_xlabel("Time (seconds)")
ax2.plot(lag, autocorr)
ax2.set_title("Autocorrelation")
ax2.set_xlabel("Lag (seconds)")
plt.tight_layout()
plt.show()