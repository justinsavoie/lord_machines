library(readr)
library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)
library(tensorflow)
library(keras)
library(stringr)
reticulate::use_python("/usr/local/Cellar/python/3.6.4_4/bin/python3.6")

print('Read data')
sysinfo <- Sys.info()

if (sysinfo[[8]]=="vpl_001"){
  setwd("~/Dropbox (Personal)/analytics_contest/")
  macro <- read_csv("campaign_data.csv")
  test <- read_csv("test_BDIfz5B.csv")
  train <- read_csv("train.csv")
}

if (sysinfo[[8]]!="vpl_001"){
  macro <- read_csv("/data/lord_machines/campaign_data.csv")
  test <- read_csv("/data/lord_machines/test_BDIfz5B.csv")
  train <- read_csv("/data/lord_machines/train.csv")
}

#train$send_date <- lubridate::dmy_hm(train$send_date)
#test$send_date <- lubridate::dmy_hm(test$send_date)
#train$hour <- lubridate::hour(train$send_date)
#test$hour <- lubridate::hour(test$send_date)
#
#train <- bind_cols(train, as.data.frame(keras::to_categorical(train$hour)[,-1]))
#test <- bind_cols(test, as.data.frame(keras::to_categorical(test$hour)[,-1]))

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

fivefreq <- findFreqTerms(dtm, 15)

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

ct <- macro%>%select(campaign_id,communication_type)%>%pull(communication_type)

ct <- to_categorical(as.numeric(factor(str_replace_all(str_to_lower(ct)," ","")))-1)
macro <- bind_cols(macro,data.frame(ct[,-1]))

train$is_click <- factor(train$is_click)
set.seed(1)
train <- train[sample(1:nrow(train),round(nrow(train)*0.3)),] 
train_train <- train %>% filter(campaign_id < 45)
train_test <- train %>% filter(campaign_id >= 45)

train_train_macro <- train_train %>% group_by(user_id)%>%
  summarise(mean_is_click=mean(as.numeric(as.character(is_click)),na.rm=TRUE)) %>%
  mutate(has_opened_one = ifelse(mean_is_click>0,1,0))

train_train <- left_join(train_train,train_train_macro)
train_test <- left_join(train_test,train_train_macro)

train_train <- left_join(train_train,(macro%>%select(campaign_id,total_links,no_of_internal_links,no_of_images,paste0("X", 2:6))))
train_test <- left_join(train_test,(macro%>%select(campaign_id,total_links,no_of_internal_links,no_of_images,paste0("X", 2:6))))

trainNB <- train_train %>% select(-is_open, 
                                  -is_click,
                                  -id, 
                                  -user_id, 
                                  -campaign_id, 
                                  -send_date,
                                  -mean_is_click,
                                  #-hour
)

testNB <- train_test %>% select(-is_open, 
                                -is_click,
                                -id, 
                                -user_id, 
                                -campaign_id, 
                                -send_date,
                                -mean_is_click,
                                #-hour
)
testNB$has_opened_one[is.na(testNB$has_opened_one)] <- 0


#system.time(classifier <- naiveBayes(trainNB, train_train$is_click, laplace = 1))
#system.time( pred <- predict(classifier, newdata=testNB) )

control <- trainControl(method="repeatedcv", number=10, repeats=3)
rf_default <- caret::train(x=trainNB,y=train_train$is_click,
                             method="rf", metric="Accuracy", 
                             trControl=control)


pred <- predict(rf_classifier,testNB)
table("Predictions"= pred,  "Actual" = train_test$is_click )

table("Predictions"= pred,  "Actual" = train_test$is_click )/
  sum(table("Predictions"= pred,  "Actual" = train_test$is_click ))

prop.table(table(train_test$is_click))
library(pROC)

pROC::roc(as.numeric(pred), as.numeric(train_test$is_click))
pROC::roc(c(1,rep(0,length(pred)-1)), as.numeric(train_test$is_click))

