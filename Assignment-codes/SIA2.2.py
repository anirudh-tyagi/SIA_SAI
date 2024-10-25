import numpy as np
import matplotlib.pyplot as plt

def generate_real_signal(N):
    """Generate a real-valued signal"""
    t = np.linspace(0, 1, N)
    # Creating a signal with multiple frequency components
    signal = 2*np.cos(2*np.pi*5*t) + 1.5*np.sin(2*np.pi*10*t)
    return signal

def analyze_dft_symmetry(signal):
    """Analyze DFT and its symmetry properties"""
    N = len(signal)
    
    # Compute DFT
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N)
    
    # Magnitude and Phase
    magnitude = np.abs(X)
    phase = np.angle(X, deg=True)
    
    # Verify symmetry property: X(N-k) = X*(k)
    symmetry_check = []
    for k in range(N):
        # Compare X(N-k) with complex conjugate of X(k)
        is_symmetric = np.allclose(X[(N-k)%N], np.conj(X[k]), atol=1e-10)
        symmetry_check.append(is_symmetric)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Original Signal
    plt.subplot(4, 1, 1)
    plt.plot(signal)
    plt.title('Original Real-Valued Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Magnitude Spectrum
    plt.subplot(4, 1, 2)
    plt.plot(freqs, magnitude)
    plt.title('Magnitude Spectrum (Symmetric)')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    # Phase Spectrum
    plt.subplot(4, 1, 3)
    plt.plot(freqs, phase)
    plt.title('Phase Spectrum (Antisymmetric)')
    plt.xlabel('Frequency')
    plt.ylabel('Phase (degrees)')
    plt.grid(True)
    
    # Real and Imaginary Parts
    plt.subplot(4, 1, 4)
    plt.plot(freqs, X.real, label='Real Part')
    plt.plot(freqs, X.imag, label='Imaginary Part')
    plt.title('Real and Imaginary Parts of DFT')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return X, symmetry_check

def main():
    # Parameters
    N = 128  # Number of samples
    
    # Generate real-valued signal
    signal = generate_real_signal(N)
    
    # Analyze DFT symmetry
    X, symmetry_check = analyze_dft_symmetry(signal)
    
    # Print mathematical verification
    print("\nDFT Symmetry Analysis:")
    print("-----------------------")
    print(f"Signal length: {N}")
    print(f"Symmetry property X(N-k) = X*(k) verified: {all(symmetry_check)}")
    
    # Demonstrate symmetry for specific frequency components
    k = 5  # Example frequency index
    print(f"\nDetailed symmetry check for k = {k}:")
    print(f"X({k}) = {X[k]:.4f}")
    print(f"X({N-k})* = {np.conj(X[N-k]):.4f}")
    print(f"Difference magnitude: {abs(X[N-k] - np.conj(X[k])):.10f}")

if __name__ == "__main__":
    main()