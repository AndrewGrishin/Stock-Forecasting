library(fracdiff);
library(forecast);
library(pracma);
library(rugarch);
pacman::p_load(progress);

wape <- function(y, y_hat) {
  return(sum(abs(y - y_hat)) / sum(abs(y)) * 100)
}

prices_not_returns <- TRUE
df <- read.csv("../Data/American Companies/AAPL 1980 2022.csv")
series <- na.omit(as.numeric(df$Open))

series <- if (prices_not_returns) series else diff(series) / series[-length(series)]
series.diff <- if (prices_not_returns) diff(series) else series

series.diff.adj <- series.diff - mean(series.diff)

test_size <- 150
series.diff.adj.train <- series.diff.adj[1:(length(series.diff) - test_size)]
series.diff.adj.test <- series.diff.adj[(length(series.diff) - test_size + 1):length(series.diff)]

forecast.values <- zeros(test_size, m = 1)

pbar_forecasting <- progress_bar$new(
  total = length(1:(test_size - 1)), 
  width = 60, 
  format = ":current/:total [:bar] :percent :eta")

for (i in 1:(test_size - 1)) {
  pbar_forecasting$tick()
  
  train_series <- c(series.diff.adj.train, if (i > 1) series.diff.adj[1:(i - 1)] else c())
  
  model.fit <- auto.arima(
    train_series, 
    seasonal = FALSE, 
    ic = c("bic"), 
    parallel = TRUE, 
    allowdrift = TRUE, 
    allowmean = TRUE,
    num.cores = 8,
    lambda = "auto")
  save.forecast <- forecast(object = model.fit, h = 1, level = 99)
  f <- save.forecast$mean[1]
  
  residuals <- model.fit$residuals
  garch_model <- ugarchspec(
    mean.model = list(armaOrder = c(0,0), arfima= FALSE, include.mean = TRUE),
    variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
    distribution.model = "norm"
  )
  
  garch_model_fit <- ugarchfit(data = residuals, spec = garch_model)
  garch_model_forecast <- ugarchforecast(fitORspec = garch_model_fit, n.ahead = 1)
  mu <- fitted(garch_model_forecast)
  sigma <- sigma(garch_model_forecast)
  
  forecast.values[i] <- f + mu + sqrt(sigma)
}
wape.value <- wape(series.diff.adj.test, forecast.values)
print(wape.value)
