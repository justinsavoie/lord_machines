library(readr)
library(dplyr)
library(caret)
library(tm)

print('Read data')
# Set it to the folder.
# setwd("~/Dropbox (Personal)/analytics_contest/")
test_macro <- read_csv("/data/lord_machines/campaign_data.csv")
test_micro <- read_csv("/data/lord_machines/test_BDIfz5B.csv")

print('Clean')
# Macro level classifier as the baseline. n=52

# Training data.
test_text <-test_macro$email_body
test_corpus <- VCorpus(VectorSource(test_text))

tdm <- DocumentTermMatrix(test_corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

train <- as.matrix(tdm)
train <- cbind(train, c(0:51))
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)

print('train')
# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

