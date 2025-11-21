import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def lowPassFilter(originalMessage, t, Wc):
    # Define parameters
    Rf = 33000
    Ri = 33000

    # Define transfer function
    num = [Wc ** 3]
    den = [1, Wc * (1 + np.sqrt(2)), Wc ** 2 * (1 + np.sqrt(2)), Wc ** 3]
    system = signal.TransferFunction(num, den)

    # Simulate the system response
    _, output, _ = signal.lsim(system, originalMessage, t)
    return output, system

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Example usage and testing
t = np.linspace(0, 10, 1000)
message = np.ones_like(t)

# Compute the frequency axis
N = len(message) - 1
Fs = 1 / (t[1] - t[0])
f = np.fft.fftshift(np.fft.fftfreq(N, 1 / Fs))
f = np.append(f, max(f) + 1)

Wc = 20  # Cutoff frequency of the filter

output = butter_lowpass_filter(message, Wc, 1000, 5)

w, h = freqz(b, a, fs=fs, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(w, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


plt.show()
