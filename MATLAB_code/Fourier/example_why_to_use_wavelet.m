clear; clc;

%% Загрузка данных

data = readtable("AAPL 1980 2022", "VariableNamingRule","preserve");
dates = data.Date;
open_prices = data.Open;
returns = diff(open_prices) ./ open_prices(1:end - 1);

%% Построение графиков

figure(1)
subplot(1, 2, 1)
plot(dates, open_prices)
title("Цена открытия акций Apple")
ylabel("Показатель цены (USD)")
xlabel("Дата")
grid on
legend("Цены: AAPL IPO - 2022", "FontSize", 15)
set(gca, "FontSize", 15)            

subplot(1, 2, 2)
plot(dates(2:end), returns * 100)
title("Доходности цены открытия акций Apple")
ylabel("Показатель доходности (%)")
xlabel("Дата")
grid on
legend("Доходности: AAPL IPO - 2022", "FontSize", 15)
set(gca, "FontSize", 15)

%% Преобразование Фурье для обоих показателей

L_prices = length(open_prices);
Fs_prices = hours(mean(diff(dates)));
f_open_prices = ((0: L_prices - 1) * Fs_prices / L_prices);

Y_open_prices = fft(open_prices);
Y_norm_open_prices = Y_open_prices * 1 / L_prices;
power_open_prices = abs(Y_norm_open_prices);

L_returns = length(returns);
Fs_returns = hours(mean(diff(dates)));
f_returns = ((0: L_returns - 1) * Fs_returns / L_returns);

Y_returns = fft(returns);
Y_norm_returns = Y_returns * 1 / L_returns;
power_returns = abs(Y_norm_returns);

%% Построение графиков амплитуд

figure(2)
subplot(1, 2, 1)
plot(f_open_prices, power_open_prices, "LineWidth", 1.25)
title("Спектр амплитуд цены открытия акций Apple")
ylabel("Амплитуда")
xlabel("Частота (Hz)")
ylim([0 0.5])
grid on
legend("Амплитуда цены: AAPL IPO - 2022", "FontSize", 15)
set(gca, "FontSize", 15)

subplot(1, 2, 2)
plot(f_returns, power_returns)
title("Спектр амплитуд доходностей акций Apple")
ylabel("Амплитуда")
xlabel("Частота (Hz)")
grid on
legend("Амплитуда доходности: AAPL IPO - 2022", "FontSize", 15)
set(gca, "FontSize", 15)
