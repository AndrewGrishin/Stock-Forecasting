library(pracma) # Hurst exponent estimation
library(fracdiff) # for arfime
library(forecast) # for arfima and forecasting
library(tseries) # for kpss test (unit root)
library(urca) # for Dickey-Fuller unit root test

# Загрузка данных
stock_prices <- read.csv("./Data/American Companies/FORD 1994 2022.csv", na.strings =  "null")
open_prices <- stock_prices$Open
open_prices <- as.numeric(na.omit(open_prices))
dates <- as.Date(stock_prices$Date, "%d.%m.%y")

# Проверка ряда на стационарность
## Аугментированный Дикки-Фулер
for (i in 0:60) {
  cat("Lag:",i , "| P value:", tseries::adf.test(open_prices, k = i)$p.value, "\n")
}
## Обыкновенный Дикки-Фуллер с выбором нужного количества лагов через Bayesian Info Criterion
summary(urca::ur.df(open_prices, type = "trend", selectlags = "BIC", lags = 60))
summary(urca::ur.df(open_prices, type = "drift", selectlags = "BIC", lags = 60))
summary(urca::ur.df(open_prices, type = "none", selectlags = "BIC", lags = 60))
## Текст KPSS
tseries::kpss.test(open_prices, null = "Level", lshort = F)
tseries::kpss.test(open_prices, null = "Trend", lshort = F)

# Переход к первым разностям
open_prices_d1 <- diff(open_prices)
dates_d1 <- dates[-1]

# Проверка ряда на стационарность
## Аугментированный Дикки-Фулер
for (i in 0:60) {
  cat("Lag:",i , "| P value:", tseries::adf.test(open_prices_d1, k = i)$p.value, "\n")
}
## Обыкновенный Дикки-Фуллер с выбором нужного количества лагов через Bayesian Info Criterion
summary(urca::ur.df(open_prices_d1, type = "trend", selectlags = "BIC", lags = 60))
summary(urca::ur.df(open_prices_d1, type = "drift", selectlags = "BIC", lags = 60))
summary(urca::ur.df(open_prices_d1, type = "none", selectlags = "BIC", lags = 60))
## Текст KPSS
tseries::kpss.test(open_prices_d1, null = "Level", lshort = F)
tseries::kpss.test(open_prices_d1, null = "Trend", lshort = F)

# рисуем первые разности
plot.ts(open_prices_d1, type = "l")

# Рисуем ACF и PACF
## Для первых разностей
acf(open_prices_d1, lag.max = 70)
pacf(open_prices_d1, lag.max = 70)
## Для исходного ряда
acf(open_prices, lag.max = 70)
pacf(open_prices, lag.max = 70)

# Вычисляем экспоненту Хёрста
hurstexp(open_prices_d1, d = 10)
summary(fracdiff(open_prices_d1))

