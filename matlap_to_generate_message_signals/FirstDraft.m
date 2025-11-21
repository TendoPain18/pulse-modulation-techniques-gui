
%%%%%%%%%%%%%%%%%%%%%****1ST TEST CASE****************%%%%%%%%%%%%%%%%%%%%%%%%
% t = 0:0.001:1;
% A=3;        %set the amplitude of carrier wave
% C=A.*sawtooth(2*pi*20*t);  % U cand adjust the freq of carrier 
% m =0.75*A.*cos(4 * pi * t);
% n = length(C);
% 
% % Modulation
% pwm = zeros(size(t)); % Initialize pwm array
% for i = 1:n
%     if m(i) >= C(i)
%         pwm(i) = 1;
%     else
%         pwm(i) = 0;
%     end
% end
% 
% %Demodulation
% demodulated_signal = zeros(size(t));
% window_size = 80; % Adjust the window size as needed
% for i = 1:n
%     start_idx = max(1, i - window_size);
%     end_idx = min(n, i + window_size);
%     demodulated_signal(i) = mean(pwm(start_idx:end_idx));
% end
% 
% % Demodulation using a low-pass filter
% % fcutoff = 50; % Choose a cutoff frequency for the low-pass filter (adjust as needed)
% % fs = 1000; % Sampling frequency
% % order = 5; % Order of the filter (adjust as needed)
% % 
% % Design a low-pass Butterworth filter
% % [b, a] = butter(order, fcutoff/(fs/2), 'low');
% % 
% % Apply the filter to the modulated signal to demodulate it
% % demodulated_signal_smooth = filtfilt(b, a, pwm);
% 
% save('message_1(t).mat', 'm', 't');
% 
% figure;
% subplot(2,1,1);
% plot(t, m);
% xlabel('Time (s)');
% ylabel('amplitude');
% title('Plot of m(t) = 3 * cos(4\pi t)');
% %U can justify the axis here 
% grid on;
% 
% subplot(2,1,2);
% plot(t,C);
% xlabel('Time (s)');
% ylabel('Amplitude');
% %U can justify the axis here
% title('carrier signal');
% grid on;
% 
% figure
% subplot(3,1,1);
% plot(t, m);
% xlabel('Time (s)');
% ylabel('amplitude');
% title('Plot of m(t) = 3 * cos(4\pi t)');
% %U can justify the axis here 
% grid on;
% 
% subplot(3,1,2);
% plot(t,pwm);
% ylim([-1.5,1.5]);
% xlabel('Time (s)');
% ylabel('Amplitude');
% %U can justify the axis here
% title('Modulated sigmal');
% grid on;
% 
% subplot(3,1,3);
% plot(t, demodulated_signal);
% xlabel('Time (s)');
% ylabel('Amplitude');
% %U can justify the axis here
% title('deModulated sigmal');
% grid on;

%%%%%%%% 2ND TEST CASE %%%%%%%%%%%%%%%%%%%%%
t = 0:0.01:8;
A = 3;  % Set the amplitude of carrier wave

% Generating a square wave carrier signal
f_carrier = 10;  % Frequency of the carrier wave
C = square(2 * pi * f_carrier * t);  % Square wave carrier signal

m = zeros(size(t));
for i = 1:length(t)
    if t(i) <= 4
        m(i) = 2 * t(i);
    end
    if t(i) > 4
        m(i) = 16 - 2 * t(i);
    end
end

% Normalize m to the range [0, 1]
m_normalized = m / max(m);

% Generate PWM signal
pwm_width = 0.5 * (1 + m_normalized); % PWM width is proportional to amplitude of m

% Modulate the PWM signal with the carrier signal
Modulated_Signal = pwm_width .* C;

save('message_2(t).mat', 'm', 't');
figure;
subplot(2, 1, 1);
plot(t, m);
xlabel('Time (s)');
ylabel('m(t)');
title('Plot of m(t) = 3 * cos(4\pi t)');
grid on;

subplot(2, 1, 2);
plot(t, C);
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-20, 20])
title('Carrier signal (Square Wave)');
grid on;

figure
subplot(2, 1, 1);
plot(t, m);
xlabel('Time (s)');
ylabel('Amplitude');
title('Plot of m(t) = 3 * cos(4\pi t)');
grid on;

subplot(2, 1, 2);
plot(t, Modulated_Signal);
xlabel('Time (s)');
ylabel('Amplitude');
title('Modulated signal (PWM)');
grid on;

