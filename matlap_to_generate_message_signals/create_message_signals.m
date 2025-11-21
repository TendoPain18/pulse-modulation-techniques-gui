t1 = 0:0.001:1;
m1 = 3 * cos(4 * pi * t1);

save('message_1(t).mat', 't1', 'm1');

figure;
plot(t1, m1);
xlabel('Time (s)');
ylabel('m(t)');
title('Plot of m(t) = 3 * cos(4\pi t)');
grid on;


t2 = 0:0.01:8;
m2 = zeros(size(t2));

for i = 1:length(t2)
    if t2(i) <= 4
        m2(i) = 2*t2(i);
    end
    if t2(i) > 4
        m2(i) = 16-2*t2(i);
    end
end
save('message_2(t).mat', 't2', 'm2');
figure;
plot(t2, m2);
xlabel('Time (s)');
ylabel('m(t)');
title('Plot of m(t) = 3 * cos(4\pi t)');
grid on;
