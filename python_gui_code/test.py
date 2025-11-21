import numpy as np


class PulseGenerator:
    def generate_pulse_signal(self, Ts, T, time):
        pulse_signal = np.zeros_like(time)
        period_length = int((len(time) / time[-1]) // (1 / Ts))
        width = int((len(time) / time[-1]) // (1 / T))
        for i in range(0, len(time), period_length):
            for j in range(width):
                if (i + j) >= len(time):
                    break
                pulse_signal[i + j] = 1
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
Ts = 0.08  # Period of the pulse train
T = 0.02  # Width of each pulse
time = np.arange(0.00, 1, 0.001)  # Time vector





cos = pulse_gen.generate_cos_signal(2, 3, 0, time)
# cos = np.zeros_like(time)
# for i in range(len(time)):
#     if time[i] <= 4:
#         cos[i] = 2*time[i]
#     if time[i] > 4:
#         cos[i] = 16 - 2 * time[i]
#
pulse_signal = pulse_gen.generate_sawtooth_pulse_signal(Ts, np.min(cos) - 1, np.max(cos) + 1, time, -1)



period_length = int((len(time) / time[-1]) // (1 / Ts))

cos_des = []
for i in range(len(cos)):
    cos_des.append(cos[i])



for i in range(0, len(cos_des), period_length):
    # value = (cos_des[i] + cos_des[i + period_length - 1]) / 2
    for j in range(period_length):
        if i+j >= len(cos_des):
            break
        cos_des[i+j] = cos_des[i]

result = np.zeros_like(time)
for i in range(len(cos)):
    result[i] = pulse_signal[i] + cos[i]

# result = pulse_gen.flat_top_modulation(cos, pulse_signal, time, Ts)

mod = []
for i in range(len(result)):
    mod.append(result[i])

for i in range(len(mod)):
    if mod[i] > (np.min(result) + np.max(result)) / 2:
        mod[i] = 1
    else:
        mod[i] = 0








def add_pulse_at_index(list, index):
    width = int((len(time) / time[-1]) // (1 / T))
    for j in range(width):
        if index-j < 0 or index-j >= len(ppm):
            continue
        ppm[index-j] = 1

def map_value(value, from_min, from_max, to_min, to_max):
    # Linear interpolation formula
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min


ppm = np.zeros_like(time)
indices = []

for i in range(len(mod) - 2, -1, 1):
    if mod[i] == 1 and mod[i+1] == 0:
        indices.append(i)

width = int((len(time) / time[-1]) // (1 / T))
for i in indices:
    period_number = i // period_length
    period_start = period_number * period_length
    period_end = (period_number + 1) * period_length
    range_start = period_start + width
    range_end = period_end
    print(i)
    print(period_number)
    print("from ", period_start, " to ", period_end)
    x = map_value(i, period_start, period_end, range_start, range_end)
    add_pulse_at_index(ppm, int(x))



# Plot the generated pulse train
import matplotlib.pyplot as plt

plt.subplot(6,1,1)
plt.plot(time, cos)
plt.plot(time, pulse_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,2)
plt.plot(time, cos_des)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,3)
plt.plot(time, pulse_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,4)
plt.plot(time, result)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,5)
plt.plot(time, mod)
for i in range(0, len(time), period_length):
    plt.axvline(x=time[i], ymin=0, ymax=1, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,6)
plt.plot(time, ppm)
for i in range(0, len(time), period_length):
    plt.axvline(x=time[i], ymin=0, ymax=1, color='r', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)


plt.show()
