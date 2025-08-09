import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

def generate_digital_signal(bits):
    """Convert a binary string to a digital signal (array of 0s and 1s)."""
    return np.array([1 if bit == '1' else 0 for bit in bits])

def add_noise(signal, noise_level=0.2):
    """Add Gaussian noise to a signal."""
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise

def calculate_snr(original, noisy):
    """Calculate Signal-to-Noise Ratio (SNR) in dB."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((noisy - original) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def calculate_ber(original, recovered):
    """Calculate Bit Error Rate (BER) by comparing original vs recovered bits."""
    recovered_binary = np.where(recovered >= 0.5, 1, 0)  # Convert to binary
    errors = np.sum(original != recovered_binary)  # Count mismatches
    return errors / len(original)

def main():
    # Get user input for bit sequence
    bits = input("Enter a digital bit sequence (e.g., 1010101): ")
    signal = generate_digital_signal(bits)
    
    # Add noise
    noisy_signal = add_noise(signal)
    
    # Apply Wiener filter for equalization
    equalized_signal = wiener(noisy_signal)

    # Calculate performance metrics
    snr = calculate_snr(signal, noisy_signal)
    ber = calculate_ber(signal, equalized_signal)
    
    # Print SNR and BER values
    print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
    print(f"Bit Error Rate (BER): {ber:.6f}")

    # Plot the signals
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(signal, 'bo-', label='Original Signal')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(noisy_signal, 'ro-', label='Noisy Signal')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(equalized_signal, 'go-', label='Equalized Signal')
    plt.legend()
    
    plt.suptitle(f'SNR: {snr:.2f} dB, BER: {ber:.6f}')
    plt.show()

if __name__ == "__main__":
    main()
