import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input
from sklearn.metrics import mean_squared_error

# Function to generate user-defined signal
def generate_signal(signal_type, frequency, samples=1000, sampling_rate=1000):
    t = np.linspace(0, 1, samples, endpoint=False)
    if signal_type == "sine":
        signal = np.sin(2 * np.pi * frequency * t)
    elif signal_type == "square":
        signal = np.sign(np.sin(2 * np.pi * frequency * t))
    elif signal_type == "sawtooth":
        signal = sawtooth(2 * np.pi * frequency * t)
    else:
        raise ValueError("Invalid signal type. Choose sine, square, or sawtooth.")
    return t, signal

# Function to add noise
def add_noise(signal, noise_level=0.2):
    noise = noise_level * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal, noise

# Create CNN-based Equalizer
def create_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(16, kernel_size=3, activation='relu', padding='same'),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(input_shape[0], activation='linear')  # Output shape matches input
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(original, noisy):
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - noisy) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Calculate Bit Error Rate (BER)
def calculate_ber(original, equalized):
    original_binary = np.where(original > 0, 1, 0)
    equalized_binary = np.where(equalized > 0, 1, 0)
    bit_errors = np.sum(original_binary != equalized_binary)
    ber = bit_errors / len(original)
    return ber

# ==== MAIN PROGRAM ====
# User input
signal_type = input("Enter signal type (sine/square/sawtooth): ").strip().lower()
frequency = float(input("Enter frequency of signal: "))

# Generate input signal
t, signal = generate_signal(signal_type, frequency)

# Add noise
noisy_signal, noise = add_noise(signal)

# Prepare data for CNN
X_train = noisy_signal.reshape(1, -1, 1)  # Reshape for CNN input
y_train = signal.reshape(1, -1)           # Target output

# Train CNN Equalizer
eq_model = create_cnn_model((X_train.shape[1], 1))
eq_model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)  # Train silently

# Predict and recover signal
equalized_signal = eq_model.predict(X_train).flatten()

# Calculate SNR and BER
snr_before = calculate_snr(signal, noisy_signal)
snr_after = calculate_snr(signal, equalized_signal)
ber = calculate_ber(signal, equalized_signal)

# Print SNR and BER
print(f"SNR Before Equalization: {snr_before:.2f} dB")
print(f"SNR After Equalization: {snr_after:.2f} dB")
print(f"Bit Error Rate (BER): {ber:.6f}")

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal, label="Original Signal", linewidth=2)
plt.title("Original Signal")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, label="Noisy Signal", color='r', linewidth=1)
plt.title(f"Noisy Signal (SNR: {snr_before:.2f} dB)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, equalized_signal, label="Equalized Signal", color='g', linewidth=2)
plt.title(f"Equalized Signal (SNR: {snr_after:.2f} dB, BER: {ber:.6f})")
plt.legend()

plt.tight_layout()
plt.show()
