library(fracdiff);
library(forecast);
library(pracma);
pacman::p_load(progress);

wape <- function(y, y_hat) {
  return(sum(abs(y - y_hat)) / sum(abs(y)) * 100)
}

get_wapes <- function(path, metrics, prices_not_returns) {
  entries <- list.files(path)
  result_dict <- c()
  
  for (file in entries) {

    company <- file
    company <- substr(company, 1, nchar(company) - 14)
    
    df <- read.csv(paste(path,file, sep = ""))
    series <- na.omit(as.numeric(df$Open))
    series <- if (prices_not_returns) series else diff(series) / series[-length(series)]
    
    series.diff <- if (prices_not_returns) diff(series) else series
    
    series.diff.adj <- series.diff - mean(series.diff)
    
    test_size = 150
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
      model.fit <- fracdiff(train_series, nar = 1, nma = 4)
      save.forecast <- forecast(object = model.fit, h = 1, level = 99)
      f <- save.forecast$mean[1]
      forecast.values[i] <- f
    }
    wape.value <- metrics(series.diff.adj.test, forecast.values)
    result_dict[company] <- wape.value
  }
  return(result_dict)
  
}

#path_us <- "../Data/American Companies/"
#us_prices <- get_wapes(path_us, wape, T)
#us_returns = get_wapes(path_us, wape, F)
#us_df <- data.frame(
#  Company = names(us_prices), 
#  "WAPE (price)" = us_prices, 
#  "WAPE (return)" = us_returns)
#colnames(us_df) <- c("Company", "WAPE (price)", "WAPE (return)")
#write.csv(us_df, "arfima_us.csv", row.names = F)

path_ch <- "../Data/Chinese Companies/"
ch_prices <- get_wapes(path_ch, wape, T)
ch_returns <-  get_wapes(path_ch, wape, F)
ch_df <- data.frame(
  Company = names(ch_prices), 
  "WAPE (price)" = ch_prices, 
  "WAPE (return)" = ch_returns)
colnames(ch_df) <- c("Company", "WAPE (price)", "WAPE (return)")
write.csv(ch_df, "arfima_ch.csv", row.names = F)


