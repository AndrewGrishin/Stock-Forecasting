%% Загрузка данных

clc; clear;
rng(42);

%data = readmatrix("data_to_denoise.csv");
%x = data(:, 1);
%y = data(:, 2);
x = linspace(-5, 5, 200);
%origin_y = 1 / 4 * x.^2 + sin(5 * x);
origin_y = 1.25 * sin(3 * x) + 2 * cos(4 * x);
y = origin_y + 2.5 * randn(size(origin_y));

L = length(y);
Fs = mean(diff(x));
f = ((0: L - 1) * Fs / L)'; % frequency samples / sec / samples = 1 / sec = Hz

figure(1)
plot(x, y, "-", "LineWidth", 1)
hold on
plot(x, origin_y, "-", "LineWidth", 1.25)
xlabel("x")
ylabel("f(x)")
grid on
legend("f(x) с шумом", "f(x) без шума", "FontSize", 14)
set(gca, "FontSize", 14)

hold off
%% Преобразование Фурье для ряда с шумом

Y = fft(y); % угол и амплитуда
Y_norm = 1/L * Y; % нормализация данных

%% Графики амплитуд и фаз
figure(2)
subplot(1, 2, 1) % 1 строка, 2 столбца, позиция текущего 1
stem(f, abs(Y_norm), "LineWidth", 1) % рисуем амплитуды
grid on
title("Спектр амплитуд")
xlabel("Частота (Hz)")
ylabel("Амплитуда")

subplot(1, 2, 2) % 1 строка, 2 столбца, позиция текущего 2
stem(f, angle(Y_norm))
grid on
title("Спектр фаз")
xlabel("частота (Hz)")
ylabel("Фаза угла")

% рисуем A * cos(2 * pi * f + phase).
% A - амплитуда, f - частота, phase - фаза.
%% Формируем таблицу данных

table1 = table(Y_norm, f', abs(Y_norm), angle(Y_norm));
table1.Properties.VariableNames = {'FFT coefs', 'Frequency', 'Amplitude', 'Phase'};

%% Восстановление ряда посредством использования k Dominant Frequencies
% Значительно влияют на наш график. Зануляем все элементы, кроме 4 с обоих
% концов, так как FFT симметричен.
% Построение статистики.

rmseVals = 0;

for k = 1:floor(L / 2)
    Y_recon = Y;
    for i = (k + 2):(L - k)
        Y_recon(i) = 0;
    end
    
    % Обратное преобразование
    
    y_clean = ifft(Y_recon);
    
    rmseVals(k) = sqrt(mean((y_clean - origin_y).^2));
end
[min_err, k_opt] = min(rmseVals);

statisticalDataRMSE = [(1:floor(L / 2))', rmseVals'];

k = k_opt;
Y_recon = Y;
for i = (k + 2):(L - k)
    Y_recon(i) = 0;
end

y_clean = ifft(Y_recon);

%writematrix(statisticalDataRMSE, "statisticalDataRMSE.csv")
%% Восстановление второй подход
% 
% k = k_opt;
% Y_recon = Y;
% for i = (k + 2):(L - k)
%     Y_recon(i) = 0;
% end

range = 1e-5:1e-7:1;


statisticalDataNewError = 0;
i = 1;
for alpha = range;
    Y_recon = Y;
    Y_recon(abs(Y_norm) < alpha) = 0;

    y_clean = ifft(Y_recon);
    
    statisticalDataNewError(i) = sqrt(mean((y_clean - origin_y).^2));
    i = i + 1;
end

[error, alpha_opt] = min(statisticalDataNewError);

Y_recon = Y;
Y_recon(abs(Y_norm) < range(alpha_opt)) = 0;
% Обратное преобразование

y_clean = ifft(Y_recon);

%% Построение финальных графиков

figure(3)
plot(x, y, "-")
hold on
plot(x, origin_y, "-", "LineWidth", 1.75)
plot(x, y_clean ,"b-", "LineWidth", 2.5)
grid on
legend("f(x) с шумом", "f(x) без шума", "f(x) очищенный", "FontSize", 20)
set(gca, "FontSize", 14)

hold off
%% Сохранение полученных результатов
denoised_data = [x', origin_y', y', y_clean', f, abs(Y_norm)', angle(Y_norm)'];
writematrix(denoised_data, "denoised_data_correct.csv")