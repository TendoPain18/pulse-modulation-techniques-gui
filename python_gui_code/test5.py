import numpy as np
from scipy.signal import butter, lfilter


class PulseGenerator:
    def generate_pulse_signal(self, Ts, T, time, mode=0, A=1):
        pulse_signal = np.zeros_like(time)
        period_length = int((len(time) / time[-1]) // (1 / Ts))
        width = int((len(time) / time[-1]) // (1 / T))
        if mode == 0:
            for i in range(0, len(time), period_length):
                for j in range(width):
                    if (i + j) >= len(time):
                        break
                    pulse_signal[i + j] = A
        else:
            for i in range(period_length, len(time), period_length):
                for j in range(width):
                    if (i - j) >= len(time):
                        break
                    pulse_signal[i - j] = A
        return pulse_signal.tolist()

    def generate_sawtooth_pulse_signal(self, Ts, a, b, time, slope=1):
        pulse_signal = np.zeros_like(time)
        period_length = int((len(time) / time[-1]) // (1 / Ts))
        amp = b - a
        step = amp / period_length
        for i in range(0, len(time), period_length):
            if slope == 1:
                current = a
            else:
                current = b
            for j in range(period_length):
                if (i + j) >= len(time):
                    break
                pulse_signal[i + j] = current
                if slope == 1:
                    current += step
                else:
                    current -= step
        return pulse_signal.tolist()

    def generate_cos_signal(self, freq, amplitude, phase, time):
        return amplitude * np.cos(2 * np.pi * freq * time + phase)

    def flat_top_modulation(self, message, pulse, time, Ts):
        modulated_signal = np.zeros_like(time)
        period_length = int(len(time) // (1 / Ts))
        for i in range(0, len(time), period_length):
            for j in range(period_length):
                if (i+j) >= len(time):
                    break
                modulated_signal[i+j] = pulse[i+j] * message[i]
        return modulated_signal




# Example usage
pulse_gen = PulseGenerator()
Ts = 0.125  # Period of the pulse train
T = 0.03125  # Width of each pulse
time = np.arange(0.00, 1, 0.001)  # Time vector





cos = pulse_gen.generate_cos_signal(2, 3, 0, time)
# cos = np.zeros_like(time)
# for i in range(len(time)):
#     if time[i] <= 4:
#         cos[i] = 2*time[i]
#     if time[i] > 4:
#         cos[i] = 16 - 2 * time[i]

pulse_signal = pulse_gen.generate_sawtooth_pulse_signal(Ts, np.min(cos) - 1, np.max(cos) + 1, time, 1)


period_length = int((len(time) / time[-1]) // (1 / Ts))

# result = pulse_gen.flat_top_modulation(cos, pulse_signal, time, Ts)

def time_domain_demodulator(signal, time, Ts, B):
    demod = np.zeros_like(signal)

    period_length = int((len(time) / time[-1]) // (1 / Ts))
    shift = 0
    for i in range(0, len(signal), period_length):

        sinc = np.zeros_like(time)
        for j in range(len(sinc)):
            temp1 = 2 * B
            temp2 = (time[j] - shift)
            temp3 = temp1 * temp2
            temp4 = np.sinc(temp3)
            sinc[j] = 2 * B * Ts * signal[i] * temp4

        demod = adder(demod, sinc)
        shift = shift + Ts

    return demod.tolist()


def add_pulse_at_index(list, index):
    width = int((len(time) / time[-1]) // (1 / T))
    for j in range(width):
        if index+j < 0 or index+j >= len(list):
            continue
        list[index+j] = 1

def map_value(value, from_min, from_max, to_min, to_max):
    # Linear interpolation formula
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min


# ppm = np.zeros_like(time)
# for i in range(len(mod)-1):
#     if cos[i] >= pulse_signal[i] and cos[i+1] < pulse_signal[i+1]:
#         add_pulse_at_index(ppm, i)



def comparator(signal_1, signal_2):
    sig = signal_1.copy()
    for i in range(len(sig)):
        if signal_1[i] > signal_2[i]:
            sig[i] = 1
        else:
            sig[i] = 0
    return sig

def regenerate_pwm(signal):
    sig = signal.copy()
    for ii in range(len(sig)):
        if sig[ii] > 0.5:
            sig[ii] = 1
        else:
            sig[ii] = 0
    return sig

def regenerate_ramp(signal, original_ramp_signal):
    sig = signal.copy()
    value = 0
    print(pulse_signal)
    for ii in range(len(sig)):
        if signal[ii] == 0:
            sig[ii] = value
        else:
            sig[ii] = signal[ii] * pulse_signal[ii]
        value = sig[ii]
    return sig

def create_refrence_pulse(siganl, pulse_signal):
    ref = pulse_gen.generate_pulse_signal(Ts, T, time, 1, np.max(pulse_signal) - np.min(pulse_signal) + 1)
    return ref

def adder(signal_1, signal_2):
    sum = signal_1.copy()
    for i in range(len(sum)):
        sum[i] = signal_1[i] + signal_2[i]

    for i in range(len(sum)-2, -1, -1):
        if sum[i] < sum[i+1] and sum[i] > np.max(regen_ramp):
            sum[i] = sum[i+1]

    return sum

def clipper(signal, clip_level):
    sig = signal.copy()
    for i in range(len(sig)):
        if sig[i] < clip_level:
            sig[i] = clip_level
    if clip_level >= 0:
        for i in range(len(sig)):
            sig[i] = sig[i] - clip_level
    else:
        for i in range(len(sig)):
            sig[i] = sig[i] + clip_level

    return sig

def get_min_max(signal):
    min = 0
    max = 0
    for i in range(len(signal)):
        if signal[i] != 0:
            min = signal[i]
            max = signal[i]
            break

    for i in signal:
        if i != 0 and i > max:
            max = i
        if i != 0 and i < min:
            min = i
    return min, max


def syncronise(signal, message):
    sig = signal.copy()
    # min, max = get_min_max(signal)

    # for i in range(len(sig)):
    #     if sig[i] != 0:
    #         sig[i] = map_value(sig[i], min, max, np.min(message), np.max(message))
    # for i in range(len(sig)-1, 0, -1):
    #     if sig[i-1] < sig[i] and sig[i-1] != 0:
    #         sig[i-1] = sig[i]

    return sig

# def adder(signal_1, signal_2):
#     sum = signal_1.copy()
#     for i in range(len(sum)-2, -1, -1):
#         if signal_2[i] > 0 and signal_2[i-1] == 0:
#             c = 0
#             while signal_2[i+c] > 0:
#                 sum[i+c] = signal_1[i] + signal_2[i]
#                 c += 1
#     return sum

def butter_lowpass_filter(data_, cutoff_, fs_):
    b_ = fs_ / 4
    c_ = 150
    a_ = (16 * c_) / fs_ ** 2
    y_ = -a_ * ((cutoff_ - b_) ** 2) + c_
    order_ = np.ceil(y_)
    frac_ = butter(order_, cutoff_, fs=fs_)
    output_ = lfilter(frac_[0], frac_[1], data_)
    return output_, frac_[0], frac_[1]

def frequency_range(sampling_frequency):
    freq_range = (0, sampling_frequency / 2)  # Frequency range from 0 to Nyquist frequency (Fs / 2)
    return freq_range


############################################## PWM #############################################
message = cos
pulse_signal = pulse_signal
mod = comparator(message, pulse_signal)
regenerate_mod = regenerate_pwm(mod)
regen_ramp = regenerate_ramp(regenerate_mod, pulse_signal)
ref = create_refrence_pulse(message, pulse_signal)
sum = adder(regen_ramp, ref)
clipped = clipper(sum, np.max(regen_ramp))
sen = syncronise(clipped, message)
fs = 1 / (time[1] - time[0])

demod = time_domain_demodulator(sen, time, Ts, 2)



# Plot the generated pulse train
import matplotlib.pyplot as plt

plt.subplot(11,1,1)
plt.plot(time, message)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11,1,2)
plt.plot(time, pulse_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11,1,3)
plt.plot(time, cos)
plt.plot(time, pulse_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11,1,4)
plt.plot(time, mod)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11,1,5)
plt.plot(time, regenerate_mod)
for i in range(0, len(time), period_length):
    plt.axvline(x=time[i], ymin=0, ymax=1, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11,1,6)
plt.plot(time, regen_ramp)
for i in range(0, len(time), period_length):
    plt.axvline(x=time[i], ymin=0, ymax=1, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11,1,7)
plt.plot(time, ref)
for i in range(0, len(time), period_length):
    plt.axvline(x=time[i], ymin=0, ymax=1, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11,1,8)
plt.plot(time, sum)
for i in range(0, len(time), period_length):
    plt.axvline(x=time[i], ymin=0, ymax=1, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11,1,9)
plt.plot(time, clipped)
for i in range(0, len(time), period_length):
    plt.axvline(x=time[i], ymin=0, ymax=1, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11,1,10)
plt.plot(time, cos)
plt.plot(time, sen)
for i in range(0, len(time), period_length):
    plt.axvline(x=time[i], ymin=0, ymax=1, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)



plt.subplot(11,1,11)
plt.plot(time, demod)
for i in range(0, len(time), period_length):
    plt.axvline(x=time[i], ymin=0, ymax=1, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

kkk = 0.01
while kkk < 1000:
    plt.pause(0.01)
    plt.clf()
    print(kkk)
    demod = time_domain_demodulator(sen, time, Ts, kkk)
    plt.subplot(11,1,11)
    plt.plot(time, demod)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    kkk += 0.01

plt.show()
