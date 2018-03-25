library(readr)
library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)

print('Read data')
sysinfo <- Sys.info()

if (sysinfo[[8]]=="vpl_001"){
  setwd("~/Dropbox (Personal)/analytics_contest/")
  macro <- read_csv("campaign_data.csv")
  test <- read_csv("test_BDIfz5B.csv")
  train <- read_csv("train.csv")
}

if (sysinfo[[8]]!="vpl_001"){
  test_macro <- read_csv("/data/lord_machines/campaign_data.csv")
  test_micro <- read_csv("/data/lord_machines/test_BDIfz5B.csv")
  train <- read_csv("/data/lord_machines/train.csv")
}



print('Clean')
# Macro level classifier as the baseline. n=52

corpus <- Corpus(VectorSource(factor(macro$email_body)))

corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus.clean)

fivefreq <- findFreqTerms(dtm, 3)

dtm.train.nb <- DocumentTermMatrix(corpus.clean, control=list(dictionary = fivefreq))

# Function to convert the word frequencies to yes (presence) and no (absence) labels
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

# Apply the convert_count function to get final training and testing DTMs
trainNB <- apply(dtm.train.nb, 2, convert_count)
trainNB <- data.frame(trainNB)
trainNB$campaign_id <- macro$campaign_id
train <- left_join(train, trainNB)

train$is_click <- factor(train$is_click)
train_train <- train %>% filter(campaign_id < 45)
train_test <- train %>% filter(campaign_id >= 45)

trainNB <- train_train %>% select(-is_open, 
                          -is_click,
                          -id, 
                          -user_id, 
                          -campaign_id, 
                          -send_date)

testNB <- train_test %>% select(-is_open, 
                                  -is_click,
                                  -id, 
                                  -user_id, 
                                  -campaign_id, 
                                  -send_date)



system.time(classifier <- naiveBayes(trainNB, train_train$is_click, laplace = 1))
system.time( pred <- predict(classifier, newdata=testNB) )
table("Predictions"= pred,  "Actual" = df.test$class )

