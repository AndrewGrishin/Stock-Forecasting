clc; clear;

%% Загрузка данных
data = rmmissing(readtable("../../Data/Chinese Companies/Kweichow Moutai 2001 2022.csv", "VariableNamingRule","preserve"));
dates_prices = data.Date;
open_prices = data.Open;
Fs_prices = hours(mean(diff(dates_prices)));

dates_returns = data.Date(2:end);
returns = diff(open_prices) ./ open_prices(1:end - 1) * 100;
Fs_returns = hours(mean(diff(dates_returns)));


%% Wavelet разложение

[wt_prices, f_prices] = cwt(open_prices, "amor", Fs_prices);
[wt_returns, f_returns] = cwt(returns, "amor", Fs_returns);
wt_prices_abs = abs(wt_prices);
wt_returns_abs = abs(wt_returns);

%% Построение scalogram

plot_scalogram_plots( ...
    dates_prices, open_prices, ...
    dates_returns, returns, ...
    dates_prices, f_prices, wt_prices_abs,...
    dates_returns, f_returns, wt_returns_abs, "Kweichow Moutai")

%% Очистка посредством Wavelet анализа

df = readtable("data_to_denoise.csv");
denoised = wdenoise(df.denoised, 8, 'DenoisingMethod', "BlockJS");
y_real = 1/4 * df.x .^ 2 + sind(5 * df.x);
plot(df.x, df.denoised)
hold on
plot(df.x, denoised)

data = [df.x, df.y, denoised];
% writematrix(data, "wavelet_example_denoising.csv")
