import numpy as np
from scipy.signal import butter, lfilter


def generate_rect_pulse_signal(Ts, T, time, mode=0, A=1):
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


def generate_sawtooth_pulse_signal(Ts, a, b, time, slope=1):
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

    # shift_steps = int((shift_angle / 360) * period_length)
    # pulse_signal = np.concatenate((pulse_signal[-shift_steps:], pulse_signal[:-shift_steps]))
    return pulse_signal


def generate_triangle_pulse_signal(Ts, a, b, time, slope=1, shift_angle=0):
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




def flat_top_modulation(message, pulse, time, Ts):
    modulated_signal = np.zeros_like(time)
    period_length = int((len(time) / time[-1]) // (1 / Ts))
    for i in range(0, len(time), period_length):
        for j in range(period_length):
            if (i + j) >= len(time):
                break
            modulated_signal[i + j] = pulse[i + j] * message[i]
    return modulated_signal.tolist()


def natural_modulation(message, pulse, time, Ts):
    modulated_signal = np.zeros_like(time)
    period_length = int((len(time) / time[-1]) // (1 / Ts))
    for i in range(0, len(time), period_length):
        for j in range(period_length):
            if (i + j) >= len(time):
                break
            modulated_signal[i + j] = pulse[i + j] * message[i + j]
    return modulated_signal.tolist()


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


def demodulate_pam_signal(signal, time, Ts, cutoff):
    fs = 1 / (time[1] - time[0])
    output = butter_lowpass_filter(signal, cutoff, fs)
    print(frequency_range(1 / Ts)[1] + 15)
    return output[0].tolist()


def normalize_vector_to_range(vector, a, b):
    # Calculate the minimum and maximum values of the vector
    min_value = np.min(vector)
    max_value = np.max(vector)

    # Normalize the vector to the given range [a, b]
    normalized_vector = a + ((vector - min_value) / (max_value - min_value)) * (b - a)

    return normalized_vector


def plot_spectrum(signal, t):
    # Compute the frequency axis
    n = len(signal) - 1
    fs = 1 / (t[1] - t[0])
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    f = np.append(f, max(f) + 1)

    # Compute the Fourier transform of the signal
    signal_spectrum = np.fft.fftshift(np.fft.fft(signal))

    # Compute the reconstructed signal
    # reconstructed_signal = np.fft.ifft(np.fft.ifftshift(signal_spectrum))

    # Compute the spectrum and the maximum used frequency index
    spectrum = signal_spectrum
    normalized_spectrum = np.abs(spectrum) / np.max(np.abs(spectrum))
    max_used_index = 1
    for i in range(len(f)):
        if f[i] >= 0:
            if np.abs(np.real(normalized_spectrum[i])) > 0.005 or np.abs(np.imag(normalized_spectrum[i])) > 0.005:
                max_used_index = f[i]

    return max_used_index

########################################################################################################################

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


def regenerate_ramp(signal, pulse_signal):
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


def create_refrence_pulse(siganl, pulse_signal, Ts, time):
    ref = generate_rect_pulse_signal(Ts, 0.02, time, 1, np.max(pulse_signal) - np.min(pulse_signal) + 1)
    return ref


def adder(signal_1, signal_2):
    sum = signal_1.copy()
    for i in range(len(sum)):
        sum[i] = signal_1[i] + signal_2[i]
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


def syncronise(signal):
    sig = signal.copy()
    for i in range(len(sig)-1, 0, -1):
        if sig[i-1] < sig[i] and sig[i-1] != 0:
            sig[i-1] = sig[i]
    return sig


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


def time_domain_demodulator_2(signal, time, Ts, B):
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


def map_value(value, from_min, from_max, to_min, to_max):
    # Linear interpolation formula
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min


def ppm_demod(data, time, Ts, A):
    band = 1
    pulse_signal = generate_triangle_pulse_signal(Ts, np.min(data) - band, np.max(data) + band, time, 1, 0)
    slope = (np.max(data) - np.min(data) + 2 * band) / (0.5 * Ts)
    period_length = int((len(time) / time[-1]) // (1 / Ts))

    mod = comparator(data, pulse_signal)

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
        if i + (j // 2) >= len(time):
            continue
        PWM_to_pam[(i + (j // 2))] = sum - 3 * shift

    normalized = PWM_to_pam.copy().tolist()

    min = np.min(normalized)
    max = np.max(normalized)

    for k in range(0, len(normalized)):
        if normalized[k] != 0:
            normalized[k] = map_value(normalized[k], 0, max, np.min(data), np.max(data))

    final = time_domain_demodulator_2(normalized, time, Ts, A)
    return final


def convert_pwm_to_pam(mod, time, period_length, band, slope, message_min, message_max):
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
        if i + (j // 2) >= len(time):
            continue
        PWM_to_pam[(i + (j // 2))] = sum - 3 * shift

    normalized = PWM_to_pam.copy()

    min = np.min(normalized)
    max = np.max(normalized)

    for k in range(0, len(normalized)):
        if normalized[k] != 0:
            normalized[k] = map_value(normalized[k], 0, max, message_min, message_max)

    return normalized


class Calculations:
    def __init__(self):
        self.graph_1_labels = []
        self.graph_1_signals = []
        self.graph_2_labels = []
        self.graph_2_signals = []
        self.graph_3_labels = []
        self.graph_3_signals = []
        self.graph_4_labels = []
        self.graph_4_signals = []

    def natural_pam_modulation(self, time, data, Ts, T, A):
        pulse_signal = generate_rect_pulse_signal(Ts, T, time)
        modulated_signal = natural_modulation(data, pulse_signal, time, Ts)

        temp = time_domain_demodulator(modulated_signal, time, Ts, A)

        cutoff = 0
        if plot_spectrum(data, time) == 2:
            cutoff = 3
        elif plot_spectrum(data, time) == 1.125:
            cutoff = 0.4
        demodulated_signal = demodulate_pam_signal(modulated_signal, time, Ts, cutoff)

        self.graph_1_labels = ["Message Signal", "time(s)", "Amplitude"]
        self.graph_1_signals = data

        self.graph_2_labels = ["Pulse Signal", "time(s)", "Amplitude"]
        self.graph_2_signals = pulse_signal

        self.graph_3_labels = ["Message VS Modulated Signals", "time(s)", "Amplitude"]
        self.graph_3_signals = [data, modulated_signal]

        self.graph_4_labels = ["Demodulated Signal", "time(s)", "Amplitude"]
        self.graph_4_signals = temp
        return time, self.graph_1_labels, self.graph_1_signals, self.graph_2_labels, self.graph_2_signals, self.graph_3_labels, self.graph_3_signals, self.graph_4_labels, self.graph_4_signals

    def flat_top_pam_modulation(self, time, data, Ts, T, A):
        pulse_signal = generate_rect_pulse_signal(Ts, T, time)
        modulated_signal = flat_top_modulation(data, pulse_signal, time, Ts)

        temp = time_domain_demodulator(modulated_signal, time, Ts, A)

        cutoff = plot_spectrum(data, time)
        if cutoff == 2:
            cutoff = 3
        elif cutoff == 1.125:
            cutoff = 0.4
        demodulated_signal = demodulate_pam_signal(modulated_signal, time, Ts, cutoff)

        self.graph_1_labels = ["Message Signal", "time(s)", "Amplitude"]
        self.graph_1_signals = data

        self.graph_2_labels = ["Pulse Signal", "time(s)", "Amplitude"]
        self.graph_2_signals = pulse_signal

        self.graph_3_labels = ["Message VS Modulated Signals", "time(s)", "Amplitude"]
        self.graph_3_signals = [data, modulated_signal]

        self.graph_4_labels = ["Demodulated Signal", "time(s)", "Amplitude"]
        self.graph_4_signals = temp
        return time, self.graph_1_labels, self.graph_1_signals, self.graph_2_labels, self.graph_2_signals, self.graph_3_labels, self.graph_3_signals, self.graph_4_labels, self.graph_4_signals

    def pwm_modulation(self, time, data, Ts, A, cutoff):
        band = 1
        pulse_signal = generate_triangle_pulse_signal(Ts, np.min(data) - band, np.max(data) + band, time, 1, 0)
        slope = (np.max(data) - np.min(data) + 2 * band) / (0.5 * Ts)
        period_length = int((len(time) / time[-1]) // (1 / Ts))


        mod = comparator(data, pulse_signal)

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
            if i + (j // 2) >= len(time):
                continue
            PWM_to_pam[(i + (j // 2))] = sum - 3 * shift

        normalized = PWM_to_pam.copy().tolist()

        min = np.min(normalized)
        max = np.max(normalized)

        for k in range(0, len(normalized)):
            if normalized[k] != 0:
                normalized[k] = map_value(normalized[k], 0, max, np.min(data), np.max(data))

        final = time_domain_demodulator_2(normalized, time, Ts, cutoff)

        self.graph_1_labels = ["Message Signal", "time(s)", "Amplitude"]
        self.graph_1_signals = data

        self.graph_2_labels = ["Pulse Signal", "time(s)", "Amplitude"]
        self.graph_2_signals = pulse_signal

        for i in range(len(mod)):
            mod[i] = mod[i] * A
        self.graph_3_labels = ["Modulated Signals", "time(s)", "Amplitude"]
        self.graph_3_signals = [["ylim", -2, A+1], mod]


        self.graph_4_labels = ["Demodulated Signal", "time(s)", "Amplitude"]
        self.graph_4_signals = final

        return time, self.graph_1_labels, self.graph_1_signals, self.graph_2_labels, self.graph_2_signals, self.graph_3_labels, self.graph_3_signals, self.graph_4_labels, self.graph_4_signals



        # data_dis = data.copy()
        # for i in range(0, len(data_dis), period_length):
        #     # value = (cos_des[i] + cos_des[i + period_length - 1]) / 2
        #     for j in range(period_length):
        #         if i + j >= len(data_dis):
        #             break
        #         data_dis[i + j] = data_dis[i]
        #
        # result = np.zeros_like(time)
        # for i in range(len(data)):
        #     result[i] = pulse_signal[i] + data[i]
        #
        # modulated_signal = result.copy().tolist()
        # for i in range(len(modulated_signal)):
        #     if modulated_signal[i] > (np.min(result) + np.max(result)) / 2:
        #         modulated_signal[i] = 1
        #     else:
        #         modulated_signal[i] = 0



    def ppm_modulation(self, time, data, Ts, T, A, cutoff):
        def add_pulse_at_index(list, index):
            width = int((len(time) / time[-1]) // (1 / T))
            for j in range(width):
                if index - j < 0 or index - j >= len(ppm):
                    continue
                ppm[index - j] = 1

        def map_value(value, from_min, from_max, to_min, to_max):
            # Linear interpolation formula
            return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
        pulse_signal_ = generate_triangle_pulse_signal(Ts, np.min(data) - 1, np.max(data) + 1, time, 1, 0)
        period_length = int((len(time) / time[-1]) // (1 / Ts))
        pulse_signal = generate_sawtooth_pulse_signal(Ts, np.min(data) - 1, np.max(data) + 1, time, 1)
        mod = comparator(data, pulse_signal)

        ppm = np.zeros_like(time).tolist()
        indices = []
        for i in range(len(mod) - 2, -1, -1):
            if mod[i] == 1 and mod[i + 1] == 0:
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

        final = ppm_demod(data, time, Ts, cutoff)


        self.graph_1_labels = ["Message Signal", "time(s)", "Amplitude"]
        self.graph_1_signals = data

        self.graph_2_labels = ["Pulse Signal", "time(s)", "Amplitude"]
        self.graph_2_signals = pulse_signal_

        for i in range(len(ppm)):
            ppm[i] = ppm[i] * A

        self.graph_3_labels = ["Modulated Signals", "time(s)", "Amplitude"]
        self.graph_3_signals = [["ylim", -2, A+1], ppm]
        for i in range(0, len(time), period_length):
            self.graph_3_signals.append(["x", time[i], 0, 1])

        self.graph_4_labels = ["Demodulated Signal", "time(s)", "Amplitude"]
        self.graph_4_signals = final

        return time, self.graph_1_labels, self.graph_1_signals, self.graph_2_labels, self.graph_2_signals, self.graph_3_labels, self.graph_3_signals, self.graph_4_labels, self.graph_4_signals



        # pulse_signal = generate_sawtooth_pulse_signal(Ts, np.min(data) - 1, np.max(data) + 1, time, -1)
        # period_length = int((len(time) / time[-1]) // (1 / Ts))
        # data_dis = data.copy()
        # for i in range(0, len(data_dis), period_length):
        #     # value = (cos_des[i] + cos_des[i + period_length - 1]) / 2
        #     for j in range(period_length):
        #         if i + j >= len(data_dis):
        #             break
        #         data_dis[i + j] = data_dis[i]
        #
        # result = np.zeros_like(time)
        # for i in range(len(data)):
        #     result[i] = pulse_signal[i] + data[i]
        #
        # modulated_signal = result.copy().tolist()
        # for i in range(len(modulated_signal)):
        #     if modulated_signal[i] > (np.min(result) + np.max(result)) / 2:
        #         modulated_signal[i] = 1
        #     else:
        #         modulated_signal[i] = 0



