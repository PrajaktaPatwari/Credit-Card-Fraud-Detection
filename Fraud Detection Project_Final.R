####################################################################
################## Import Libraries ################################
####################################################################

library(caTools)        #split data - sample
library(caret)          #for predict function
library(ROSE)           #ROC curve - check once again
library(randomForest)   
library(e1071)          #SVM
library(corrplot)
library(xgboost)
library(DMwR)
library(pROC)
library(flexclust)
####################################################################
################## Load the file ###################################
####################################################################

frauddetection = read.csv('projectdataF18.csv',header=TRUE,stringsAsFactors=FALSE)
str(frauddetection)
summary(frauddetection)

####################################################################
################## Check for Imbalance dataset  ####################
####################################################################

table(frauddetection$Fake)
116/(89884+116)   #data highly imbalanced as 99.8% as not fake
89884/(89884+116)

####################################################################
#################  Visualization ###################################
###################################################################

#Plot for distribution of fake variable:
Fake_plot = ggplot(frauddetection,aes(x=Fake))+
  geom_bar(position ="dodge",fill="darkred")+
  scale_y_continuous()+
  scale_x_discrete()+
  ggtitle("Unbalanced Fake varaible")
Fake_plot

#Plot for distribution of Amount variable:
Amount_plot = ggplot(frauddetection,aes(x=log(Amount)))+  #normal distribution
  geom_histogram(aes(y=..density..),binwidth=.25,
                 colour="black", fill="white")+
  geom_density(alpha =0.2,adjust=0.25,fill="#FF6666")+
  ggtitle("Distribution of Amount")
Amount_plot


#Fake vs. Time:
#Noraml plot:
plot(frauddetection$Fake,frauddetection$Time)

#Using ggplot:
library(ggplot2)
p = ggplot(frauddetection,aes(x=Fake,y=Time))+geom_point(color="purple")
p


####################################################################
########################## Data exploration ########################
################## To get columns with NA values ###################
####################################################################
#checking for missing values
missing_values = sort(colSums(is.na(frauddetection)>0), decreasing = T) 
missing_values
#No missing values

####################################################################
################## Correlation #####################################
####################################################################
cor(frauddetection)  # making the correlation plot. All the variables look not correlated.
corrplot(cor(frauddetection[,1:30]), method = "number")

####################################################################
################## Normalize the amount variable ###################
####################################################################

frauddetection$Amount = scale(frauddetection$Amount)  # Normalizing only amount 
#as all the other variables are normalized

####################################################################
##################### Balance the dataset ##########################
##################### Applying SMOTE method ########################
####################################################################

#Now we are going to balance the data set using smote

# Use 10-fold cross-validation
ctrl <- trainControl(method = "cv",
                     number = 10,
                     verboseIter = T,
                     classProbs = T,
                     sampling = "smote",  
                     summaryFunction = twoClassSummary,
                     savePredictions = T)



####################################################################
##################### Split the dataset ############################
####################################################################

set.seed(3000)
split = sample.split(frauddetection$Fake,SplitRatio = 0.7) 
frauddetection <- frauddetection[,-1] #removing time variable as no sufficient data is available to analyze time variable
train_set = subset(frauddetection,split==TRUE)
test_set= subset(frauddetection,split==FALSE)
str(frauddetection)
str(train_set)
str(test_set)
Fake_train =  subset(frauddetection$Fake,split==TRUE)  # Subsetting only the fake column of train
Fake_test = subset(frauddetection$Fake,split==FALSE)  # Subsetting only the fake column of test


####################################################################
###### Finding unwanted variables using var importance function#####
####################################################################

train_set$Fake = as.factor(train_set$Fake)
levels(train_set$Fake) <- make.names(c(0, 1))
RF_model = randomForest(Fake ~., data = train_set,ntree=100) #running rf model to use in the varImp function

importance(RF_model)
varImpPlot(RF_model,type=2) # plotting varImp plot
# From the graph we see that the variabes X3,X13,X20,X23,X24,X27,X28 are less important, but 
# removing these did not increase the accuracy. 
# So none of the variables were removed for the below modeling.



#####################################################################
#############  Model 1 : Logistic Regression #########################
#####################################################################

#Logistic model:
logistic_model = glm(formula = Fake ~., family = "binomial" , data = train_set)
summary(logistic_model)

#Predicting and using various thresholds 
#threshold > 0.2
table(Fake_test, as.numeric(predict(logistic_model, test_set, type = "response") > 0.2))
(26960+25)/(26960+25+15)
#0.9994444

#threshold > 0.5
table(Fake_test, as.numeric(predict(logistic_model, test_set, type = "response") > 0.5))
(26961+25)/(26961+14+25)
#0.9994815

#threshold > 0.99
table(Fake_test, as.numeric(predict(logistic_model, test_set, type = "response") > 0.99))
(26965+25)/(26965+25+10)

#Plotting ROC curve:
roc_logistic = roc(Fake_test, as.numeric(predict(logistic_model, test_set, type = "response")))
plot(roc_logistic, main = paste0("AUC: ", round(pROC::auc(roc_logistic), 3)))
#AUC=97.6


#Validation:

newdataset = read.csv("projectdataF18+validation.csv")
str(newdataset)
newdataset = newdataset[,-1]  #removing time variable
newdataset$Amount = scale(newdataset$Amount)  # Normalizing only amount 


logistic_model_Validation <- predict(logistic_model, newdata = newdataset)
logistic_model_Validation = ifelse(logistic_model_Validation >= 0.5,1,0)

table(logistic_model_Validation)


######################################################################
#############  Model 2 : Random Forest Regression#####################
#####################################################################

set.seed(3000)
split = sample.split(frauddetection$Fake,SplitRatio = 0.7) 
str(frauddetection)
train_set = subset(frauddetection,split==TRUE)
test_set= subset(frauddetection,split==FALSE)
str(frauddetection)
str(train_set)
str(test_set)
Fake_train =  subset(frauddetection$Fake,split==TRUE)  # Subsetting only the fake column of train
Fake_test = subset(frauddetection$Fake,split==FALSE)

train_set$Fake = as.factor(train_set$Fake)
levels(train_set$Fake) <- make.names(c(0, 1))
RF_model <- train(Fake ~., data = train_set, method = "rf", trControl = ctrl, verbose = T, metric = "ROC")

prediction <- predict(RF_model, test_set, type = "prob")
table(as.numeric(prediction$X1 > 0.5), Fake_test)
(26751+28)/(26751+28+214+7)  #0.9918 accuracy

roc_RF <- roc(Fake_test, predict(RF_model, test_set, type = "prob")$X1)
plot(roc_RF, main = paste0("AUC: ", round(pROC::auc(roc_RF), 3))) #Plotting ROC curve
#AUC=0.986 which is better than logistic

#Validation:

newdataset = read.csv("projectdataF18+validation.csv")
str(newdataset)
newdataset = newdataset[,-1]  #removing time variable
newdataset$Amount = scale(newdataset$Amount)  # Normalizing only amount 
randomForest_model_Validation <- predict(logistic_model, newdata = newdataset)

randomForest_model_Validation = ifelse(randomForest_model_Validation >= 0.5,1,0)

table(randomForest_model_Validation)

######################################################################
#############  Model 4 : XGboost #####################################
#####################################################################

set.seed(3000)                                           
split = sample.split(frauddetection$Fake,SplitRatio = 0.7)
train_set = subset(frauddetection,split==TRUE)
test_set= subset(frauddetection,split==FALSE)
str(frauddetection)
Fake_train =  subset(frauddetection$Fake,split==TRUE)  # Subsetting only the fake column of train
Fake_test = subset(frauddetection$Fake,split==FALSE)
str(train_set)


#we need to remove 31st column of fake/not observation
train_xg  = xgb.DMatrix(data = as.matrix(train_set[,-30]), label = as.numeric(train_set$Fake))
test_xg = xgb.DMatrix(data = as.matrix(test_set[,-30]), label = as.numeric(test_set$Fake))

xg_boost = xgboost(data = train_xg,nrounds = 125 , gamma = 0.1, max_depth = 2, objective = "binary:logistic", nthread = 2)

xgb.importance(model = xg_boost)   #Important variables with their freq

prediction_xgb = predict(xg_boost, test_xg) #predicting using xgboost model
table(as.numeric(prediction_xgb > 0.5), Fake_test)
(26965+35)/(26965+35) # 100% accuracy so far the best model
(26963+26)/(26963+26+2+9) #0.9995926
roc_xg_boost = roc(Fake_test, prediction_xgb)
plot(roc_xg_boost, main = paste0("AUC: ", round(pROC::auc(roc_xg_boost), 3)))  #The AUC = 1, 
#which is the max/bestpossible result for the test data 

#Validation using the xgboost model

newdataset = read.csv("projectdataF18+validation.csv") #reading the validation data set
str(newdataset)
newdataset = newdataset[,-1]
newdataset$Amount = scale(newdataset$Amount)# removing time variable
newdataset = as.matrix(newdataset) #converting to matrix as our xgboost model is in matrix form

xgbpred <- predict(xg_boost,newdataset)
xgbpred = ifelse(xgbpred >= 0.5,1,0)
table(xgbpred)
#####################################################################
#############  Cluster and then predict medthod: #########################
#####################################################################

#Running few lines of previous data

frauddetection = read.csv('projectdataF18.csv',header=TRUE,stringsAsFactors=FALSE)
frauddetection=frauddetection[,-1]
frauddetection$Amount = scale(frauddetection$Amount)  # Normalizing only amount 
ctrl <- trainControl(method = "cv",
                     number = 10,
                     verboseIter = T,
                     classProbs = T,
                     sampling = "smote",  
                     summaryFunction = twoClassSummary,
                     savePredictions = T)



str(frauddetection)  
set.seed(3000)
split = sample.split(frauddetection$Fake,SplitRatio = 0.7)
train_set = subset(frauddetection,split==TRUE)
test_set= subset(frauddetection,split==FALSE)
d <- dist(frauddetection, method = "euclidean") # finding distance to get dendogram
fit <- hclust(d, method="ward") 
plot(fit) # Due to the lage size of data we get an error, dendogram cannot be plotted

#pre processing data set before clustering

limitedTrain = train_set
limitedTrain$Fake = NULL
limitedTest = test_set
limitedTest$Fake = NULL
str(limitedTrain)


preproc = preProcess(limitedTrain)
normTrain = predict(preproc, limitedTrain)
normTest = predict(preproc, limitedTest)
library(flexclust)
set.seed(144)
k=3   # Assuming k value, as dendogram cannot be plottted. Will vary and find the best k value


km = kmeans(normTrain, centers = k) # using kmeans to cluster data based on their 
#distances with the centroid of each cluster
km.kcca = as.kcca(km, normTrain)
clusterTrain = predict(km.kcca)
clusterTest = predict(km.kcca, newdata=normTest)

#### clustering data into 3 clusters ###########
# clustering data into 3 sets and from each cluster forming a train and a test variable
fakeTrain1=subset(train_set,clusterTrain==1) 
fakeTrain2=subset(train_set,clusterTrain==2)
fakeTrain3=subset(train_set,clusterTrain==3)
fakeTest1=subset(test_set,clusterTest==1)
fakeTest2=subset(test_set,clusterTest==2)
fakeTest3=subset(test_set,clusterTest==3)

##################################################################################
#############  Applying different models to each cluster ########################
##################################################################################

## Using glm models to each cluster

Model1=glm(Fake~.,data=fakeTrain1, family=binomial)
Model2=glm(Fake~.,data=fakeTrain2, family=binomial)
Model3=glm(Fake~.,data=fakeTrain3, family=binomial)

##### Prediction of test set using the logistic models 1,2 and 3 ######

PredictTest1 = predict(Model1, newdata=fakeTest1, type="response")
table(fakeTest1$Fake,PredictTest1>=0.5)

PredictTest2 = predict(Model2, newdata=fakeTest2, type="response")
table(fakeTest2$Fake,PredictTest2>=0.5)

PredictTest3 = predict(Model3, newdata=fakeTest3, type="response")
table(fakeTest3$Fake,PredictTest3>=0.5)
##################################################################################
centroidOfClusters=km$center  #Finding cluster centroid to calculate
#distance from the validation test manually

write.csv(centroidOfClusters,file=paste("centroidOfClusters.csv"))

#After finding centroid, each data point from the validation data was alloted to either 
#one of the 3 clusters based on their euclidean distances. For eg, the data point was 
# sent to cluster1 is it has the least distance wrt cluster1. Please refer excel.

##################################################################################
#############  Applying clustering to validation data set ########################
##################################################################################

####### CLustering validation dataset ####################
ValDataset = read.csv("Validation_Clusters.csv") #reading the clustered validation data set
#ValDataset$Amount = scale(ValDataset$Amount) # scaling the amount variable
ValDataset=ValDataset[,-1] #remove time variable

str(ValDataset)  # Note that the validation file has one extra column clus, manually 
# inserted. clus=1 means that observation belongs to cluster 1, 
# if M=2 the observation belongs to cluster 2 and so on.

ValDataset1 = subset(ValDataset,clus==1) #Grouping validation data sets into 3 sets 
#based on their distances from cluster
ValDataset2 = subset(ValDataset,clus==2) 
ValDataset3 = subset(ValDataset,clus==3)  
ValDataset11=ValDataset1[,-30] #removing the extra column clus as it is no longer needed for modeling
ValDataset22=ValDataset2[,-30]  
ValDataset33=ValDataset3[,-30]    

str(ValDataset11)  #checking if the clus variable was deleted
str(ValDataset22)
str(ValDataset33)

ValDataset11$Amount=as.matrix(ValDataset11$Amount)
ValDataset22$Amount=as.matrix(ValDataset22$Amount)
ValDataset33$Amount=as.matrix(ValDataset33$Amount)

##### Prediction of validation set using the above models ######

PredictValDataset1 = predict(Model1, newdata=ValDataset11, type="response")


PredictValDataset2 = predict(Model2, newdata=ValDataset22, type="response")


PredictValDataset3 = predict(Model3, newdata=ValDataset33, type="response")


########### Putting together all the predicted values  ##########################
AllPredictions = c(PredictValDataset1>=0.5, PredictValDataset2>=0.5,PredictValDataset3>=0.5)
table(AllPredictions)

prediction_glm=write.csv(AllPredictions,file = paste("AllPredictionsxgbglm.csv"))






