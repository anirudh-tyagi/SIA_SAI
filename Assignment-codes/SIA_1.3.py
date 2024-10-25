import numpy as np
import matplotlib.pyplot as plt

def deconvolve(y, h, lambda_reg=1e-3):
    Y = np.fft.fft(y)
    H = np.fft.fft(h, len(y))
    X = Y / (H + lambda_reg)  # Regularized division
    x = np.real(np.fft.ifft(X))
    return x

# Create original signal
t = np.linspace(0, 1, 1000)
x_original = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

# Create impulse response (e.g., moving average filter)
h = np.ones(50) / 50

# Convolve x with h to get y
y = np.convolve(x_original, h, mode='same')

# Perform deconvolution
x_deconvolved = deconvolve(y, h)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
ax1.plot(t, x_original)
ax1.set_title("Original Signal")
ax2.plot(t, y)
ax2.set_title("Convolved Signal (y = x * h)")
ax3.plot(t, x_deconvolved[:len(t)])
ax3.set_title("Deconvolved Signal")
plt.tight_layout()
plt.show()