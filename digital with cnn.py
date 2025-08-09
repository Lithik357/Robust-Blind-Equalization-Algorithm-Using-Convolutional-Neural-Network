import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input

# Function to add Gaussian noise
def add_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

# Function to calculate SNR
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean((signal - noise) ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Function to calculate Bit Error Rate (BER)
def calculate_ber(original, equalized):
    original_bits = np.where(original > 0, 1, 0)  # Convert signal to binary (1s and 0s)
    equalized_bits = np.where(equalized > 0, 1, 0)  # Convert equalized signal to binary
    errors = np.sum(original_bits != equalized_bits)  # Count bit mismatches
    return errors / len(original)

# User inputs the signal
def get_user_signal():
    print("Enter at least 5 signal values (comma-separated, e.g., 0.1, 0.3, 0.5, ...): ")
    user_input = input().strip()
    signal = np.array([float(x) for x in user_input.split(",")])
    if len(signal) < 5:
        print("Error: Please enter at least 5 values.")
        exit()
    return signal

# Get user-defined signal
original_signal = get_user_signal()
signal_length = len(original_signal)

# User inputs the SNR value
snr_db = float(input("Enter the SNR value (in dB, e.g., 10): "))

# Add noise to the signal
noisy_signal = add_noise(original_signal, snr_db)

# Reshape for CNN input
X_train = np.array([add_noise(original_signal, snr_db) for _ in range(1000)]).reshape(-1, signal_length, 1)
y_train = np.array([original_signal for _ in range(1000)]).reshape(-1, signal_length, 1)

# Define CNN model
model = Sequential()
model.add(Input(shape=(signal_length, 1)))  # ✅ Fixed Input Layer Warning
model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding="same"))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding="same"))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(signal_length, activation='linear'))  

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Predict equalized signal
noisy_signal_reshaped = noisy_signal.reshape(1, signal_length, 1)
equalized_signal = model.predict(noisy_signal_reshaped, verbose=0).flatten()

# Calculate SNR and BER
calculated_snr = calculate_snr(original_signal, noisy_signal)
bit_error_rate = calculate_ber(original_signal, equalized_signal)

# Print the results
print(f"Calculated SNR after equalization: {calculated_snr:.2f} dB")
print(f"Bit Error Rate (BER): {bit_error_rate:.6f}")

# Plot the signals
plt.figure(figsize=(10, 5))
plt.plot(original_signal, label="Original Signal", linestyle="dashed", color='green')
plt.plot(noisy_signal, label="Noisy Signal", linestyle="dotted", color='red')
plt.plot(equalized_signal, label="Equalized Signal", color='blue')
plt.legend()
plt.title("Blind Equalization using CNN")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
