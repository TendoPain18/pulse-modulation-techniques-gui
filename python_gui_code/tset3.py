import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from scipy.fft import fft


def butter_lowpass_filter(data_, cutoff_, fs_):
    b_ = fs_ / 4
    c_ = 45
    a_ = (16 * c_) / fs_ ** 2
    y_ = -a_ * ((cutoff_ - b_) ** 2) + c_
    order_ = np.ceil(y_)
    frac_ = butter(order_, cutoff_, fs=fs_)
    output_ = lfilter(frac_[0], frac_[1], data_)
    print(type(output_))
    return output_, frac_[0], frac_[1]


fm = 10
fc = 1000
fs = int(fc / 0.005)
print("Fs: " + str(fs))
cutoff = 1

# Generate a unit impulse signal
t = np.linspace(0, 10, fs)
message = np.cos(2 * np.pi * fm * t)
carrier = np.cos(2 * np.pi * fc * t)
modulated = message * carrier
demod = modulated * carrier
demod_filtered = butter_lowpass_filter(demod, cutoff, fs)


while cutoff < fs / 2 - 21:
    cutoff += 700
    print(cutoff)
    frac = butter_lowpass_filter(demod, cutoff, fs)
    w, h = freqz(frac[1], frac[2], fs=fs, worN=1000)

    plt.subplot(6, 1, 1)
    plt.plot(w, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    plt.subplot(6, 1, 2)
    plt.plot(t, carrier, 'b')
    plt.xlabel('Carrier')
    plt.xlim([0, 8 / fc])
    plt.grid()

    plt.subplot(6, 1, 3)
    plt.plot(t, modulated, 'b')
    plt.xlabel('Modulated')
    plt.xlim([0, 8 / fm])
    plt.grid()

    plt.subplot(6, 1, 4)
    plt.plot(t, demod, 'b')
    plt.xlabel('Demodulated')
    plt.xlim([0, 8 / fm])
    plt.grid()

    plt.subplot(6, 1, 5)
    plt.plot(t, message, 'b')
    plt.xlabel('Message')
    plt.xlim([0, 8 / fm])
    plt.grid()

    plt.subplot(6, 1, 6)
    plt.plot(t, frac[0], 'b')
    plt.xlabel('Filtered')
    plt.xlim([0, 8 / fm])
    plt.grid()

    plt.subplots_adjust(hspace=0.35)
    plt.pause(0.01)
    plt.clf()

plt.show()
