library(doParallel)
detectCores()
cl <- makeCluster(4)
registerDoParallel(cl)
getDoParWorkers()
library(caret)
library(mlbench)
library(rpart)
library(RUnit)
library(plotly)
library(C50)
library(kernlab)
install.packages("inum")
library(kknn)



iphone_matrix<- read.csv("/Users/jdduk/Desktop/smallmatrix_labeled_8d/iphone_smallmatrix_labeled_8d.csv")
str(iphone_matrix)
summary(iphone_matrix)
plot_ly(iphone_matrix, x= iphone_matrix$iphonesentiment, type='histogram')
is.na(iphone_matrix)
sum(is.na(iphone_matrix))

#Factor iPhonesentiment
df<- iphone_matrix
str(df)
df$iphonesentiment <- as.factor(df$iphonesentiment)
df$iphonesentiment <- as.character(df$iphonesentiment)

plot(df$iphonesentiment, type="h", col="blue")
plot(df2$galaxysentiment, type= "h", col="orange")
ggplot(df2, aes(x=galaxysentiment, y= count))+
  geom_text(size =3)

#Preprocessing
cor(iphone_matrix)
corrplot(cor(iphone_matrix))
corrplot(cor(galaxy_matrix))

nzvMetrics <- nearZeroVar(df, saveMetrics = TRUE)
nzvMetrics
nzv <- nearZeroVar(df, saveMetrics = FALSE)
nzv
# create a new data set and remove near zero variance features
iphoneNZV <- df[,-nzv]
str(iphoneNZV)


#Recursive Feature Elimination
#Sample data b4 using RFE
set.seed(123)
dfSample<- df[sample(1:nrow(df), 1000, replace = FALSE),]

#rfeControl
rfectrl <- rfeControl(functions = rfFuncs,
                   method= "repeatedcv",
                   repeats=1,
                   verbose= FALSE)
#Use rfe and omit the response variable
rfeResults <- rfe(dfSample[,1:58], 
                  dfSample$iphonesentiment,
                  sizes=(1:58),
                  rfeControl=rfectrl)
rfeResults
plot(rfeResults, type=c("g","o"))

#create new data set with rfe recommended features
iphoneRFE <- df[,predictors(rfeResults)]
iphoneRFE$iphonesentiment <- df$iphonesentiment
str(iphoneRFE)

# Split data into testing and training
set.seed(123)
train_index <- createDataPartition(iphoneNZV$iphonesentiment,
                                   # 70% for training
                                   p = .70,
                                   times = 1,
                                   list = FALSE)
iphoneNZVTrain <- iphoneNZV[ train_index, ] # Training
iphoneNZVTest  <- iphoneNZV[-train_index, ] # Testing
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#C5.0
iphoneNZV_C5 <- train(form = iphonesentiment~.,
                 data = iphoneNZVTrain,
                 method = 'C5.0',
                 trControl = ctrl, # Cross-validation
                 tuneLength = 10)
iphoneNZV_C5
iphoneNZV_C5_pred <- predict(iphoneNZV_C5, iphoneNZVTest)
iphoneNZV_C5_pred
postResample(iphoneNZV_C5_pred, iphoneNZVTest$iphonesentiment)
# Create a confusion matrix from C5 predictions 
cmC5 <- confusionMatrix(iphoneNZV_C5_pred, iphoneNZVTest$iphonesentiment) 
cmC5

#RandomForest
iphoneNZV_RF <- train(form = iphonesentiment~.,
                      data = iphoneNZVTrain,
                      method = 'rf',
                      trControl = ctrl, # Cross-validation
                      tuneLength = 10)
iphoneNZV_RF
iphoneNZV_RF_pred <- predict(iphoneNZV_RF, iphoneNZVTest)
iphoneNZV_RF_pred
postResample(iphoneNZV_RF_pred, iphoneNZVTest$iphonesentiment)
# Create a confusion matrix from random forest predictions 
cmRF <- confusionMatrix(iphoneNZV_RF_pred, iphoneNZVTest$iphonesentiment) 
cmRF

#SVM
iphoneNZV_SVM <- train(form = iphonesentiment~.,
                       data = iphoneNZVTrain,
                       method = 'svmLinear3',
                       trControl = ctrl, # Cross-validation
                       tuneLength = 10)
iphoneNZV_SVM
iphoneNZV_SVM_pred <- predict(iphoneNZV_SVM, iphoneNZVTest)
iphoneNZV_SVM_pred
postResample(iphoneNZV_SVM_pred, iphoneNZVTest$iphonesentiment)
# Create a confusion matrix from SVM predictions 
cmSVM <- confusionMatrix(iphoneNZV_SVM_pred, iphoneNZVTest$iphonesentiment) 
cmSVM

#kknn
iphoneNZV_kknn <- train(form = iphonesentiment~.,
                        data = iphoneNZVTrain,
                        method = 'kknn',
                        trControl = ctrl, # Cross-validation
                        tuneLength = 10)
iphoneNZV_kknn
iphoneNZV_kknn_pred <- predict(iphoneNZV_kknn, iphoneNZVTest)
iphoneNZV_kknn_pred
postResample(iphoneNZV_kknn_pred, iphoneNZVTest$iphonesentiment)
# Create a confusion matrix from SVM predictions 
cmkknn <- confusionMatrix(iphoneNZV_kknn_pred, iphoneNZVTest$iphonesentiment) 
cmkknn


#Import large matrix
iphone_largematrix<- read.csv("/Users/jdduk/Desktop/Helio_Sentiment_Analysis_Matrix_Data/iphoneLargeMatrix.csv")


nzvlmMetrics <- nearZeroVar(iphone_largematrix, saveMetrics = TRUE)
nzvlmMetrics
nzvlm <- nearZeroVar(iphone_largematrix, saveMetrics = FALSE)
nzvlm
# create a new data set and remove near zero variance features
iphonelmNZV <- iphone_largematrix[,-nzvlm]
str(iphonelmNZV)

#FinalPrediction
finalpred <- predict(iphoneNZV_RF, iphone_largematrix)
finalpred
summary(finalpred)



























galaxy_matrix<- read.csv("/Users/jdduk/Desktop/smallmatrix_labeled_8d/galaxy_smallmatrix_labeled_9d.csv")
str(galaxy_matrix)
summary(galaxy_matrix)
plot_ly(galaxy_matrix, x= galaxy_matrix$galaxysentiment, type='histogram')
is.na(galaxy_matrix)
sum(is.na(galaxy_matrix))

#Factor iPhonesentiment
df2<- galaxy_matrix
str(df2)
df2$galaxysentiment <- as.factor(df2$galaxysentiment)


#Preprocessing
cor(galaxy_matrix)
corrplot(cor(galaxy_matrix))

nzvgalaxy <- nearZeroVar(df2, saveMetrics = FALSE)
nzvgalaxy
# create a new data set and remove near zero variance features
galaxyNZV <- df2[,-nzvgalaxy]
str(galaxyNZV)

# Split data into testing and training
set.seed(123)
train_indexgalaxy <- createDataPartition(galaxyNZV$galaxysentiment,
                                   # 70% for training
                                   p = .70,
                                   times = 1,
                                   list = FALSE)
galaxyNZVTrain <- galaxyNZV[ train_indexgalaxy, ] # Training
galaxyNZVTest  <- galaxyNZV[-train_indexgalaxy, ] # Testing
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#C5.0
galaxyNZV_C5 <- train(form = galaxysentiment~.,
                      data = galaxyNZVTrain,
                      method = 'C5.0',
                      trControl = ctrl, # Cross-validation
                      tuneLength = 10)
galaxyNZV_C5
galaxyNZV_C5_pred <- predict(galaxyNZV_C5, galaxyNZVTest)
galaxyNZV_C5_pred
postResample(galaxyNZV_C5_pred, galaxyNZVTest$galaxysentiment)
# Create a confusion matrix from C5 predictions 
cmgC5 <- confusionMatrix(galaxyNZV_C5_pred, galaxyNZVTest$galaxysentiment) 
cmgC5

#RandomForest
galaxyNZV_RF <- train(form = galaxysentiment~.,
                      data = galaxyNZVTrain,
                      method = 'rf',
                      trControl = ctrl, # Cross-validation
                      tuneLength = 10)
galaxyNZV_RF
galaxyNZV_RF_pred <- predict(galaxyNZV_RF, galaxyNZVTest)
galaxyNZV_RF_pred
postResample(galaxyNZV_RF_pred, galaxyNZVTest$galaxysentiment)
# Create a confusion matrix from random forest predictions 
cmgRF <- confusionMatrix(galaxyNZV_RF_pred, galaxyNZVTest$galaxysentiment) 
cmgRF

#SVM
galaxyNZV_SVM <- train(form = galaxysentiment~.,
                       data = galaxyNZVTrain,
                       method = 'svmLinear3',
                       trControl = ctrl, # Cross-validation
                       tuneLength = 10)
galaxyNZV_SVM
galaxyNZV_SVM_pred <- predict(galaxyNZV_SVM, galaxyNZVTest)
galaxyNZV_SVM_pred
postResample(galaxyNZV_SVM_pred, galaxyNZVTest$galaxysentiment)
# Create a confusion matrix from SVM predictions 
cmgSVM <- confusionMatrix(galaxyNZV_SVM_pred, galaxyNZVTest$galaxysentiment) 
cmgSVM

#kknn
galaxyNZV_kknn <- train(form = galaxysentiment~.,
                        data = galaxyNZVTrain,
                        method = 'kknn',
                        trControl = ctrl, # Cross-validation
                        tuneLength = 10)
galaxyNZV_kknn
galaxyNZV_kknn_pred <- predict(galaxyNZV_kknn, galaxyNZVTest)
galaxyNZV_kknn_pred
postResample(galaxyNZV_kknn_pred, galaxyNZVTest$galaxysentiment)
# Create a confusion matrix from SVM predictions 
cmgkknn <- confusionMatrix(galaxyNZV_kknn_pred, galaxyNZVTest$galaxysentiment) 
cmgkknn


#Import large matrix
galaxy_largematrix<- read.csv("/Users/jdduk/Desktop/Helio_Sentiment_Analysis_Matrix_Data/galaxyLargeMatrix.csv")


nzvglmMetrics <- nearZeroVar(galaxy_largematrix, saveMetrics = TRUE)
nzvglmMetrics
nzvglm <- nearZeroVar(galaxy_largematrix, saveMetrics = FALSE)
nzvglm
# create a new data set and remove near zero variance features
galaxylmNZV <- galaxy_largematrix[,-nzvglm]
str(galaxylmNZV)

#FinalPrediction
finalpredg <- predict(galaxyNZV_RF, galaxy_largematrix)
finalpredg
summary(finalpredg)

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)

plot(finalpred, finalpredg)



