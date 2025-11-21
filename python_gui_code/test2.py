import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parameters
Fs = 1000  # Sampling frequency
T = 1      # Duration of the signal in seconds
t = np.arange(0, T, 1/Fs)  # Time vector

# Original message (cosine signal)
f_message = 10  # Frequency of the message signal
message_signal = np.cos(2 * np.pi * f_message * t)

# Modulate the message signal with a higher frequency cosine to produce replicas
f_carrier = 250  # Carrier frequency (higher than the message frequency)
modulated_signal = np.cos(2 * np.pi * f_carrier * t) * message_signal

demod_signal = np.cos(2 * np.pi * f_carrier * t) * message_signal * np.cos(2 * np.pi * f_carrier * t)

# Define a sharp low-pass filter
def sharp_lowpass_filter(data, cutoff_frequency, fs):
    order = 60  # Filter order
    b, a = butter(order, cutoff_frequency / (fs / 2), btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data * 2

# Apply sharp low-pass filter to recover the original message
cutoff_frequency = f_carrier * 2.1  # Choose cutoff frequency (higher than message frequency)


recovered_signal = sharp_lowpass_filter(modulated_signal * np.cos(2 * np.pi * f_carrier * t), cutoff_frequency, Fs)

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.title('Original Message Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, modulated_signal, label='Modulated Signal')
plt.title('Modulated Signal with Replicas')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, demod_signal, label='Demod Signal')
plt.title('Recovered Signal before Filtering')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t, recovered_signal, label='Recovered Signal')
plt.title('Recovered Signal after Filtering')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.ylim([-1, 1])
plt.legend()

plt.tight_layout()
plt.show()
