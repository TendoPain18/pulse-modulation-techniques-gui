import numpy as np
import matplotlib.pyplot as plt

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

    def generate_sawtooth_pulse_signal(self, Ts, a, b, time, slope=1, shift_angle=0):
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

        shift_steps = int((shift_angle / 360) * period_length)
        pulse_signal = np.concatenate((pulse_signal[-shift_steps:], pulse_signal[:-shift_steps]))

    def generate_triangle_pulse_signal(self, Ts, a, b, time, slope=1, shift_angle=0):
        Ts = 0.5*Ts
        pulse_signal = np.zeros_like(time)
        period_length = int(np.ceil((len(time) / time[-1]) / (1 / Ts)))
        print(len(time) / period_length)
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
            slope = slope * -1

        shift_steps = int((shift_angle / 360) * period_length)
        pulse_signal = np.concatenate((pulse_signal[-shift_steps:], pulse_signal[:-shift_steps]))


        return pulse_signal.tolist()

    def generate_cos_signal(self, freq, amplitude, phase, time):
        return amplitude * np.cos(2 * np.pi * freq * time + phase)

    def flat_top_modulation(self, message, pulse, time, Ts):
        modulated_signal = np.zeros_like(time)
        period_length = int(len(time) // (1 / Ts))
        for i in range(0, len(time), period_length):
            for j in range(period_length):
                if (i + j) >= len(time):
                    break
                modulated_signal[i + j] = pulse[i + j] * message[i]
        return modulated_signal


pulse_gen = PulseGenerator()
Ts = 0.4  # Period of the pulse train
T = 0.1  # Width of each pulse
time = np.arange(0.00, 8, 0.01)  # Time vector


def comparator(signal_1, signal_2):
    sig = signal_1.copy()
    for i in range(len(sig)):
        if signal_1[i] > signal_2[i]:
            sig[i] = 1
        else:
            sig[i] = 0
    return sig


# cos = pulse_gen.generate_cos_signal(2, 3, 0, time)
cos = np.zeros_like(time)
for i in range(len(time)):
    if time[i] <= 4:
        cos[i] = 2*time[i]
    if time[i] > 4:
        cos[i] = 16 - 2 * time[i]


band = 1
pulse_signal = pulse_gen.generate_triangle_pulse_signal(Ts, np.min(cos)-band, np.max(cos)+band, time, 1, 0)
slope = (np.max(cos) - np.min(cos) + 2*band) / (0.5*Ts)

period_length = int((len(time) / time[-1]) // (1 / Ts))


##############################################################################################

def flat_top_modulation(message, pulse, time, Ts):
    modulated_signal = np.zeros_like(time)
    period_length = int((len(time) / time[-1]) // (1 / Ts))
    for i in range(0, len(time), period_length):
        for j in range(period_length):
            if (i + j) >= len(time):
                break
            modulated_signal[i + j] = pulse[i + j] * message[i]
    return modulated_signal.tolist()


pulse_signal_pam_FT = pulse_gen.generate_pulse_signal(Ts, T, time)
modulated_signal = flat_top_modulation(cos, pulse_signal_pam_FT, time, Ts)


##############################################################################################

def map_value(value, from_min, from_max, to_min, to_max):
    # Linear interpolation formula
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min


def adder(signal_1, signal_2):
    sum = signal_1.copy()
    for i in range(len(sum)):
        sum[i] = signal_1[i] + signal_2[i]
    return sum


def time_domain_demodulator(signal, time, Ts, B):
    demod = np.zeros_like(signal)

    period_length = int((len(time) / time[-1]) // (1 / Ts))
    shift = 0
    for i in range(0, len(signal)):
        if signal[i] != 0:
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


message = cos
pulse_signal = pulse_signal
mod = comparator(message, pulse_signal)


def convert_pwm_to_pam():
    PWM_to_pam = np.zeros_like(mod)
    for i in range(0, len(time), period_length):
        sum = 0
        x = 0
        for j in range(period_length):
            if i + j >= len(time):
                continue
            if mod[i + j] == 1:
                x = j
            sum += mod[i + j]

        shift = np.ceil(abs(band / slope) * (1 / (time[1] - time[0])))
        print(shift)
        PWM_to_pam[(i + (j // 2))] = sum - 3 * shift



PWM_to_pam = np.zeros_like(mod)
for i in range(0, len(time), period_length):
    sum = 0
    x = 0
    for j in range(period_length):
        if i + j >= len(time):
            continue
        if mod[i + j] == 1:
            x = j
        sum += mod[i + j]

    shift = np.ceil(abs(band / slope) * (1/(time[1]-time[0])))
    print(shift)
    PWM_to_pam[(i + (j//2))] = sum - 3*shift








normalized = PWM_to_pam.copy()

min = np.min(normalized)
max = np.max(normalized)


for k in range(0, len(normalized)):
    if normalized[k] != 0:
        normalized[k] = map_value(normalized[k], 0, max, 0, 8)



final = time_domain_demodulator(normalized, time, Ts, 3)




##############################################################################################


plt.subplot(11, 1, 1)
plt.plot(time, message)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11, 1, 2)
plt.plot(time, pulse_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11, 1, 3)
plt.plot(time, cos)
plt.plot(time, pulse_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11, 1, 4)
plt.plot(time, mod)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11, 1, 5)
plt.plot(time, modulated_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11, 1, 6)
plt.plot(time, PWM_to_pam)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11, 1, 7)
plt.plot(time, cos)
plt.plot(time, normalized)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(11, 1, 8)
plt.plot(time, final)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.show()

kkk = 0.01
while kkk < 1000:
    plt.pause(0.01)
    plt.clf()
    print(kkk)
    demod = time_domain_demodulator(normalized, time, Ts, kkk)
    plt.subplot(1, 1, 1)
    plt.plot(time, cos)
    plt.plot(time, demod)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    kkk += 0.01

plt.show()
