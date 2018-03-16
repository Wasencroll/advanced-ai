## ==================== AI - MACHINE LEARNING ASSIGNMENT ==================== ##

## Loading libraries
library(readr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(sparklyr)

## Setting working directory
setwd("C:/Users/alcroll/OneDrive - Deloitte (O365D)/Documents/R workspace/AI")

## Reading input file (red wines) as ;-delimited file
weather_raw <- read_csv("GlobalLandTemperatures-GlobalTemperatures.csv")

## Cleaning dataset
## Average temperature per year
weather <- weather_raw %>%
  filter(!is.na(LandAverageTemperature)) %>%
  select(dt, LandAverageTemperature) %>%
  mutate(dt = as.integer(substring(dt, 1,4))) %>%
  group_by(dt) %>%
  summarize(avg_temperature = mean(LandAverageTemperature)) %>%
  rename(year = dt)

## Plotting the observations of average yearly temperature from 1750-today
ggplot(weather, aes(x = year, y = avg_temperature)) +
  geom_line(col = "black",  lwd = 0.5) + 
  geom_point(col = "black") +
  theme_bw() + 
  ggtitle("Yearly Average Temperature")
ggsave("yearly_average.png")

## Creating random indexes for weather normalized dataset
## 20% of weather dataset
indexes <- sample(1:nrow(weather), size = 0.2*nrow(weather))

## Creating test and training datasets
## Test set will have 20% of original dataset
## Training set will have 80%of original dataset
test <- weather[indexes,]
train <- weather[-indexes,]

## Fitting regression models of polynomials 1,2,3
fit1 <- lm(avg_temperature ~ year, data = train)
fit2 <- lm(avg_temperature ~ poly(year, 2, raw = TRUE), data = train)
fit3 <- lm(avg_temperature ~ poly(year, 3, raw = TRUE), data = train)

## Plotting regression models
ggplot(weather, aes(x = year, y = avg_temperature, color = "Observation")) +
  geom_point() + 
  geom_line() +
  geom_smooth(method = "lm", se = FALSE, formula = y ~ x, aes(color = "Polynomial1")) +
  geom_smooth(method = "lm", se = FALSE, formula = y ~ poly(x, 2, raw = TRUE), aes(color = "Polynomial2")) + 
  geom_smooth(method = "lm", se = FALSE, formula = y ~ poly(x, 3, raw = TRUE), aes(color = "Polynomial3")) +
  ggtitle("Fitting Regression Models of Polynomial = [1,2,3]") +
  scale_color_manual(name = "Polynomial Regression", 
                     values = c(Polynomial1 = "navy", Polynomial2 = "red", 
                                Polynomial3 = "green4", Observation = "black")) +
  theme_bw()
ggsave("regression-models1+2+3.png")

## Bias-variance tradeoff
## Initializing tradeoff data frame with columns polynomial, bias, variance, error
tradeoff_all <- data.frame(polynomial = numeric(0), model = numeric(0), bias = numeric(0), variance = numeric(0))

## Bias Function
f_bias_new <- function(x){
  sum(abs(predict(lm(avg_temperature ~ poly(year, k, raw = TRUE), data = x),
                  newdata = x)/nrow(x) - x$avg_temperature/nrow(x)))
}

## Variance Function
## Building model for entire dataset
f_variance_new <- function(x, y){
  sum(abs(predict(lm(avg_temperature ~ poly(year, k, raw = TRUE), data = x), newdata = y)/nrow(x) - 
        predict(lm(avg_temperature ~ poly(year, k, raw = TRUE), data = x), newdata = x)/nrow(x)))
}

# K-folds validation
weatherKFolds <- weather[sample(nrow(weather)),]

#Create 10 folds
kfolds <- cut(seq(1, nrow(weatherKFolds)), breaks = 10, labels = FALSE)

#Perform 10 fold cross validation
tradeoff_all <- tradeoff_all[0,]
for(k in 1:20){
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(kfolds==i, arr.ind=TRUE)
    testData <- weatherKFolds[testIndexes, ]
    trainData <- weatherKFolds[-testIndexes, ]
    tradeoff_all[nrow(tradeoff_all)+1,] <- c(k, i, f_bias_new(trainData), f_variance_new(trainData, testData),0)
  }
}

tradeoff <- tradeoff[0,]
tradeoff <- tradeoff_all %>%
  group_by(polynomial) %>%
  select(polynomial, bias, variance) %>%
  summarize(bias = mean(bias),
            variance = mean(variance))

## Plotting bias and variance as complexity increases
ggplot(tradeoff, aes(x = polynomial)) + 
  geom_line(aes(y = bias, color = "Bias")) +
  geom_point(aes(y = bias, color = "Bias")) + theme_bw() +
  geom_line(aes(y = variance, color = "Variance")) +
  geom_point(aes(y = variance, color = "Variance")) +
  ggtitle("Bias-Variance tradeoff") + 
  xlab("Polynomial") +
  ylab("") +
  scale_color_manual(name = "Bias vs. Variance", values = c(Bias = "navy", Variance = "green4"))
ggsave("bias-variance-tradeoff.png")

## Plotting regression model
ggplot(weather, aes(x = year, y = avg_temperature, color = "Observation")) +
  geom_point() + 
  geom_line() +
  geom_smooth(method = "lm", se = FALSE, formula = y ~ poly(x, 6, raw = TRUE), aes(color = "Polynomial6")) +
  ggtitle("Fitting Regression Model of Polynomial = 6") +
  theme_bw() +
  scale_color_manual(name = "Model Fit", values = c(Observation = "black", Polynomial6 = "red"))
ggsave("regression-model6.png")

## ==================== END ====================