% Parameters
fm = 2; % Carrier frequency
fc = 20; % Message frequency
fs = 1000; % Sampling frequency
t = 1;


n = 0:1/fs:t;
n = n(1:end-1);


duty = 10;
per = fs/fc;
on_t = per/duty;

s = square(2*pi*fc*n, duty);




% Message Signal
m = sin(2*pi*fm*n);

% Triangular wave
A = 1.25;
c = A.*sawtooth(2*pi*fc*n);   % Carrier sawtooth

list = {};
for i = 1:length(c)-1
    if c(i) == -1 && c(i+1) == 1
        list{end+1} = i+1;
    end
end
list = [{1}, list];

% Find IDs where carrier is greater than message
id = find(c > m);
idd = diff(id);
iddd = find(idd ~=1);
temp(1) = id(1);
temp(2:length(iddd)+1) = id(iddd+1);
ppm = zeros(1,length(n));
% PPM Signal
for i = 1:length(temp)
    ppm(temp(i): temp(i) + on_t - 1) = 1;
end

% Demodulation
% Detecting pulse positions
pulse_positions = find(ppm == 1);
demodulated_signal = m(pulse_positions);







% Plotting
subplot(4,1,1); plot(n,m,'LineWidth',2); title('Message Signal'); hold on;
subplot(4,1,2); plot(n,s,'LineWidth',2); title('Pulse Train'); grid on; ylim([-0.2 1.2]);

subplot(4,1,3);
plot(n,ppm,'LineWidth',2, 'Color', 'r');
hold on
title('PPM Signal'); grid on;
for i = 1:length(list)
    value = n(list{i});
    line([value, value], ylim, 'Color', 'k', 'LineWidth', 2);
    hold on
end

subplot(4,1,4); stem(pulse_positions, demodulated_signal, 'r', 'LineWidth', 0.5); title('Demodulated Message Signal'); grid on;


