library(rugarch)
df <- na.omit(read.csv("../Data/American Companies/AAPL 1980 2022.csv"))
prices_not_returns <- F

prices <- if (prices_not_returns) diff(df$Open) else diff(df$Open) / df$Open[1:length(df$Open) - 1]

modelspec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 1)),
  distribution.model = "std"
)

modelfit <- ugarchfit(modelspec, prices)
infocriteria(modelfit)

forecast <- ugarchforecast(modelfit, n.ahead= 1)
print(forecast@forecast@seriesFor + forecast@forecast@sigmaFor^2)
print(prices[length(prices)])