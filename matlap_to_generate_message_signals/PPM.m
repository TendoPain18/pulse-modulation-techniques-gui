fm = 2;
fc = 20;
fs = 1000;
t = 1;
n = 0:1/fs:t;
n = n(1:end-1);
duty = 10;
% no. of samples in one square wave period
per = fs/fc;
on_t = per/duty;

s = square(2*pi*fc*n, duty);
list = {};
for i = 1:length(s)-1
    if s(i) == -1 && s(i+1) == 1
        list{end+1} = i+1;
    end
end
list = [{1}, list];

% Message Signal
m = sin(2*pi*fm*n);

% Triangular wave
A = 1.25;
c = A.*sawtooth(2*pi*fc*n);   % Carrier sawtooth


id = find(c > m);
idd = diff(id);
iddd = find(idd ~=1);
temp(1) = id(1);
temp(2:length(iddd)+1) = id(iddd+1);
ppm = zeros(1,length(s));

for i = 1:length(temp)
    ppm(temp(i): temp(i) + on_t - 1) = 1;
end

subplot(3,1,1); plot(n,m,'LineWidth',2); title('Message Signal'); hold on;
subplot(3,1,2); plot(n,s,'LineWidth',2); title('Pulse Train'); grid on;
subplot(3,1,3);
plot(n,ppm,'LineWidth',2, 'Color', 'r');
hold on
title('PPM Signal'); grid on;
for i = 1:length(list)
    value = n(list{i});
    line([value, value], ylim, 'Color', 'k', 'LineWidth', 2);
    hold on
end




