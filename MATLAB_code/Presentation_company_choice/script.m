clear; clc;

%% Загрузка данных

sinopec = rmmissing(readtable( ...
    "../../Data/Chinese Companies/Kweichow Moutai 2001 2022.csv", ...
    "VariableNamingRule", "preserve"));
ge = rmmissing(readtable( ...
    "../../Data/American Companies/Coca Cola 1962 2022.csv", ...
    "VariableNamingRule", "preserve"));

sinopec_prices_dates = sinopec.Date;
sinopec_prices = sinopec.Open;
sinopec_returns_dates = sinopec_prices_dates(2:end);
sinopec_returns = diff(sinopec_prices) ./ sinopec_prices(1:end - 1);

ge_prices_dates = ge.Date;
ge_prices = ge.Open;
ge_returns_dates = ge_prices_dates(2:end);
ge_returns = diff(ge_prices) ./ ge_prices(1:end - 1);

clear ge sinopec

%% Визуализация данных

us_and_china_plot( ...
    sinopec_prices_dates, sinopec_prices, ...
    sinopec_returns_dates, sinopec_returns * 100, ...
    ge_prices_dates, ge_prices, ...
    ge_returns_dates, ge_returns * 100, "Coca Cola", "Kweichow Moutai")