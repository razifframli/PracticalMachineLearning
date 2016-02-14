#The following Libraries were used and should be installed for this project.
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(e1071)

set.seed(12345)#Finally, load the same seed with the following line of code:

#First load the data to memory solely
trainingData<-read.csv("pml-training.csv")
testingData<-read.csv("pml-testing.csv")

#Devided traning dataset into two, 60% for 
#Partioning Training data set into two data sets, 60% for Training, 40% for Testing:

startTrain <- createDataPartition(y=trainingData$classe, p=0.6, list=FALSE)
Training <- trainingData[startTrain, ] 
Testing <- trainingData[-startTrain, ]
dim(Training)
dim(Testing)

#The following transformations were used to clean the data:                                                            
#1st Cleaning: Clean NearZeroVariance (NZV) Variables and run this code to view possible NZV Variables:

DataNZV <- nearZeroVar(Training, saveMetrics=TRUE)

NZVvars <- names(Training) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
                                  "kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
                                  "max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
                                  "var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
                                  "stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
                                  "kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
                                  "max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
                                  "kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
                                  "skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
                                  "amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
                                  "skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
                                  "max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
                                  "amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
                                  "avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
                                  "stddev_yaw_forearm", "var_yaw_forearm")

Training <- Training[!NZVvars]

#To check the new N?? of observations
dim(Training)

#2nd Cleaning: Drop first column of Dataset which is ID variable. So that it does not interfer with ML Algorithms.
Training <- Training[c(-1)]

#3th Cleaning: Cleaning Variables which is more than a 60% threshold of NAs.
trainingV3 <- Training #creating another subset to iterate in loop
for(i in 1:length(Training)) { #for every column in the training dataset
  if( sum( is.na( Training[, i] ) ) /nrow(Training) >= .6 ) { #if n?? NAs > 60% of total observations
    for(j in 1:length(trainingV3)) {
      if( length( grep(names(Training[i]), names(trainingV3)[j]) ) ==1)  { #if the columns are the same:
        trainingV3 <- trainingV3[ , -j] #Remove that column
      }   
    } 
  }
}

#To check the new N?? of observations
dim(trainingV3)

#Seting back to our set:
Training <- trainingV3
rm(trainingV3)

clean1 <- colnames(Training)
clean2 <- colnames(Training[, -58]) #already with classe column removed
Testing <- Testing[clean1]
testingData <- testingData[clean2]

#To check the new N?? of observations
dim(Testing)

#To check the new N?? of observations
dim(testingData)

for (i in 1:length(testingData) ) {
  for(j in 1:length(Training)) {
    if( length( grep(names(Training[i]), names(testingData)[j]) ) ==1)  {
      class(testingData[j]) <- class(Training[i])
    }      
  }      
}

#And to make sure Coertion really worked, simple smart ass technique:
testingData <- rbind(Training[2, -58] , testingData) #row 2 does not mean anything, this will be removed right now.
testingData <- testingData[-1,]

#Using Machine Learning algorithm for prediction: Decision Tree
modFitA1 <- rpart(classe ~ ., data=Training, method="class")
fancyRpartPlot(modFitA1)# to view the decision tree with fancy run this command.

predictionsA1 <- predict(modFitA1, Testing, type = "class")
confusionMatrix(predictionsA1, Testing$classe)#Using confusion Matrix to test results:

#Using Machine Learning algorithm for prediction: Random Forests
modFitB1 <- randomForest(classe ~., data=Training)

predictionsB1 <- predict(modFitB1, Testing, type = "class")#Predicting in-sample error
confusionMatrix(predictionsB1, Testing$classe)#Using confusion Matrix to test results

#Random Forests yielded better Results, as expected where the accuracy of Random Forest algorithm is 0.999 while Decision Tree algorithm is 0.879.

#Generating Files to submit as answers for the Assignment:
predictionsB2 <- predict(modFitB1, testingData, type = "class")
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("test cases_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictionsB2)

