# GET & SET WD

getwd()
setwd('/content/LoanTest/LoanTest/LoanTest')

#Read from CSV

test<-read.csv('test.csv',header=TRUE,sep=',')
train<-read.csv('train.csv',header=TRUE,sep=',')

#Statistics

install.packages('psych',dep=TRUE)
install.packages('Hmisc',dep=TRUE)
install.packages('e1071', dep=TRUE)

library(psych)
library(Hmisc)
describe(train)
describe(test)
names(train)
names(test)
summary(train)
summary(test)

# /* MASTER DATA PREP START */

#Missing Value Check & Str Check - TRAIN

str(train$UniqueID)
str(train$ltv)
str(train$branch_id)
str(train$State_ID)
str(train$supplier_id)
str(train$manufacturer_id)
str(train$Current_pincode_ID)
names(train)
str(train$Age)
str(train$Employment.Type)
str(Aadhar_flag)
str(train$MobileNo_Avl_Flag)
str(Acct.Age.Num)
str(Credit.History.Length.Num)
train$branch_id<-as.factor(train$branch_id)
train$State_ID<-as.factor(train$State_ID)
train$supplier_id<-as.factor(train$supplier_id)
train$manufacturer_id<-as.factor(train$manufacturer_id)
train$Current_pincode_ID<-as.factor(train$Current_pincode_ID)
sum(is.na(train$Employment.Type))
mode(train$Employment.Type)


attach(train)


MobileNo_Avl_Flag<-as.factor(MobileNo_Avl_Flag)
Aadhar_flag<-as.factor(Aadhar_flag)
PAN_flag<-as.factor(PAN_flag)
VoterID_flag<-as.factor(VoterID_flag)
Driving_flag<-as.factor(Driving_flag)
Passport_flag<-as.factor(Passport_flag)


#attach(test)
#Missing Value Check & Str Check - TEST


str(test$UniqueID)
str(test$ltv)
str(test$branch_id)
str(test$State_ID)
str(test$supplier_id)
str(test$manufacturer_id)
str(test$Current_pincode_ID)
names(test)
str(test$Age)
str(test$Employment.Type)
str(Aadhar_flag)
str(test$MobileNo_Avl_Flag)
str(Acct.Age.Num)
str(Credit.History.Length.Num)
test$branch_id<-as.factor(test$branch_id)
test$State_ID<-as.factor(test$State_ID)
test$supplier_id<-as.factor(test$supplier_id)
test$manufacturer_id<-as.factor(test$manufacturer_id)
test$Current_pincode_ID<-as.factor(test$Current_pincode_ID)
sum(is.na(test$Employment.Type))
test$MobileNo_Avl_Flag<-as.factor(test$MobileNo_Avl_Flag)
test$Aadhar_flag<-as.factor(test$Aadhar_flag)
test$PAN_flag<-as.factor(test$PAN_flag)
test$VoterID_flag<-as.factor(test$VoterID_flag)
test$Driving_flag<-as.factor(test$Driving_flag)
test$Passport_flag<-as.factor(test$Passport_flag)

# Data Transformation - TRAIN

train$Scaled_Disbursed<-log(train$disbursed_amount)
train$Scaled_AssetCost<-log(train$asset_cost)
train$ScaledLTV<-train$ltv/10
train$Scaled_Primary_Num_Accts<-sqrt(train$PRI.NO.OF.ACCTS)
train$Scaled_Primary_Active_Accts<-sqrt(train$PRI.ACTIVE.ACCTS)
train$Scaled_Primary_Overdue_Accts<-sqrt(train$PRI.OVERDUE.ACCTS)
train$Scaled_Primary_Curr_Bal<-(train$PRI.CURRENT.BALANCE)+1
train$New_Scaled_Primary_Curr_Bal<-log(train$Scaled_Primary_Curr_Bal)
train$Scaled_Primary_Sanct_Amt<-(train$PRI.SANCTIONED.AMOUNT)+1
train$New_Scaled_Primary_Sanct_Amt<-log(train$Scaled_Primary_Sanct_Amt)
train$Scaled_Primary_Disbursed_Amt<-(train$PRI.DISBURSED.AMOUNT)+1
train$New_Scaled_Primary_Disbursed_Amt<-log(train$Scaled_Primary_Disbursed_Amt)
train$Scaled_Secondary_Num_Accts<-sqrt(train$SEC.NO.OF.ACCTS)
train$Scaled_Secondary_Active_Accts<-sqrt(train$SEC.ACTIVE.ACCTS)
train$Scaled_Secondary_Overdue_Accts<-sqrt(train$SEC.OVERDUE.ACCTS)
train$Scaled_Secondary_Curr_Bal<-(train$SEC.CURRENT.BALANCE)+1
train$New_Scaled_Secondary_Curr_Bal<-log(train$Scaled_Secondary_Curr_Bal)
train$Scaled_Secondary_Sanct_Amt<-(train$SEC.SANCTIONED.AMOUNT)+1
train$New_Scaled_Secondary_Sanct_Amt<-log(train$Scaled_Secondary_Sanct_Amt)
train$Scaled_Secondary_Disbursed_Amt<-(train$SEC.DISBURSED.AMOUNT)+1
train$New_Scaled_Secondary_Disbursed_Amt<-log(train$Scaled_Secondary_Disbursed_Amt)
train$Scaled_Primary_Installment_Amount<-(train$PRIMARY.INSTAL.AMT)+1
train$New_Scaled_Primary_Installment_Amount<-log(train$Scaled_Primary_Installment_Amount)
train$Scaled_Secondary_Installment_Amount<-(train$SEC.INSTAL.AMT)+1
train$New_Scaled_Secondary_Installment_Amount<-log(train$Scaled_Secondary_Installment_Amount)
train$Scaled_New_Accts_in_6_Mo<-sqrt(train$NEW.ACCTS.IN.LAST.SIX.MONTHS)
train$Scaled_Delinquent_Accts_in_7_Mo<-sqrt(train$DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS)
sum(is.na(train$New_Scaled_Primary_Curr_Bal))
train$New_Scaled_Primary_Curr_Bal<-ifelse(is.na(train$New_Scaled_Primary_Curr_Bal),0,train$New_Scaled_Primary_Curr_Bal)
sum(is.na(train$New_Scaled_Secondary_Curr_Bal))
train$New_Scaled_Secondary_Curr_Bal<-ifelse(is.na(train$New_Scaled_Secondary_Curr_Bal),0,train$New_Scaled_Secondary_Curr_Bal)
write.csv(train, file="Train_interim.csv")

# Data Transformation - TEST

test$Scaled_Disbursed<-log(test$disbursed_amount)
test$Scaled_AssetCost<-log(test$asset_cost)
test$ScaledLTV<-test$ltv/10
test$Scaled_Primary_Num_Accts<-sqrt(test$PRI.NO.OF.ACCTS)
test$Scaled_Primary_Active_Accts<-sqrt(test$PRI.ACTIVE.ACCTS)
test$Scaled_Primary_Overdue_Accts<-sqrt(test$PRI.OVERDUE.ACCTS)
test$Scaled_Primary_Curr_Bal<-(test$PRI.CURRENT.BALANCE)+1
test$New_Scaled_Primary_Curr_Bal<-log(test$Scaled_Primary_Curr_Bal)
test$Scaled_Primary_Sanct_Amt<-(test$PRI.SANCTIONED.AMOUNT)+1
test$New_Scaled_Primary_Sanct_Amt<-log(test$Scaled_Primary_Sanct_Amt)
test$Scaled_Primary_Disbursed_Amt<-(test$PRI.DISBURSED.AMOUNT)+1
test$New_Scaled_Primary_Disbursed_Amt<-log(test$Scaled_Primary_Disbursed_Amt)
test$Scaled_Secondary_Num_Accts<-sqrt(test$SEC.NO.OF.ACCTS)
test$Scaled_Secondary_Active_Accts<-sqrt(test$SEC.ACTIVE.ACCTS)
test$Scaled_Secondary_Overdue_Accts<-sqrt(test$SEC.OVERDUE.ACCTS)
test$Scaled_Secondary_Curr_Bal<-(test$SEC.CURRENT.BALANCE)+1
test$New_Scaled_Secondary_Curr_Bal<-log(test$Scaled_Secondary_Curr_Bal)
test$Scaled_Secondary_Sanct_Amt<-(test$SEC.SANCTIONED.AMOUNT)+1
test$New_Scaled_Secondary_Sanct_Amt<-log(test$Scaled_Secondary_Sanct_Amt)
test$Scaled_Secondary_Disbursed_Amt<-(test$SEC.DISBURSED.AMOUNT)+1
test$New_Scaled_Secondary_Disbursed_Amt<-log(test$Scaled_Secondary_Disbursed_Amt)
test$Scaled_Primary_Installment_Amount<-(test$PRIMARY.INSTAL.AMT)+1
test$New_Scaled_Primary_Installment_Amount<-log(test$Scaled_Primary_Installment_Amount)
test$Scaled_Secondary_Installment_Amount<-(test$SEC.INSTAL.AMT)+1
test$New_Scaled_Secondary_Installment_Amount<-log(test$Scaled_Secondary_Installment_Amount)
test$Scaled_New_Accts_in_6_Mo<-sqrt(test$NEW.ACCTS.IN.LAST.SIX.MONTHS)
test$Scaled_Delinquent_Accts_in_7_Mo<-sqrt(test$DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS)
sum(is.na(test$New_Scaled_Primary_Curr_Bal))
test$New_Scaled_Primary_Curr_Bal<-ifelse(is.na(test$New_Scaled_Primary_Curr_Bal),0,test$New_Scaled_Primary_Curr_Bal)
sum(is.na(test$New_Scaled_Secondary_Curr_Bal))
test$New_Scaled_Secondary_Curr_Bal<-ifelse(is.na(test$New_Scaled_Secondary_Curr_Bal),0,test$New_Scaled_Secondary_Curr_Bal)
sum(is.na(test$New_Scaled_Primary_Sanct_Amt))
test$New_Scaled_Primary_Sanct_Amt<-ifelse(is.na(test$New_Scaled_Primary_Sanct_Amt),0,test$New_Scaled_Primary_Sanct_Amt)


sum(is.na(train$Scaled_New_Primary_Curr_Bal))
names(train)
train$Scaled_New_Primary_Curr_Bal<-ifelse(is.na(train$Scaled_New_Primary_Curr_Bal),0,train$Scaled_New_Primary_Curr_Bal)
sum(is.na(train$New_Scaled_Secondary_Curr_Bal))


# /* MASTER DATA PREP END */

# Check for Significant Variables by running GLM - 78.2%

LogReg<-glm(loan_default~Scaled_Disbursed+Scaled_AssetCost+ScaledLTV+Scaled_Primary_Num_Accts+
              Scaled_Primary_Active_Accts+Scaled_Primary_Overdue_Accts+
              New_Scaled_Primary_Sanct_Amt+New_Scaled_Primary_Disbursed_Amt+Scaled_Secondary_Num_Accts+
              Scaled_Secondary_Active_Accts+Scaled_Secondary_Overdue_Accts+
              New_Scaled_Secondary_Sanct_Amt+New_Scaled_Secondary_Disbursed_Amt+New_Scaled_Primary_Installment_Amount+
              New_Scaled_Secondary_Installment_Amount+Age+Employment.Type+State_ID+MobileNo_Avl_Flag+Aadhar_flag+
              PAN_flag+VoterID_flag+Driving_flag+Passport_flag+PERFORM_CNS.SCORE+Acct.Age.Num+Credit.History.Length.Num+
              NO.OF_INQUIRIES+NEW.ACCTS.IN.LAST.SIX.MONTHS+DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS,data=train,family='binomial')
plot(LogReg)

train$YPredict<-predict.glm(LogReg,train,type='response')
View(train)
train$Loan_Status_Predicted<-ifelse(train$YPredict>=0.5,1,0)
table(train$Loan_Status_Predicted)

#Confusion Matrix - GLM - 78.2%

table(train$Loan_Status_Predicted,train$loan_default)

library(e1071)
SVcL <- svm(loan_default~Scaled_Disbursed+Scaled_AssetCost+ScaledLTV+Scaled_Primary_Num_Accts+
             Scaled_Primary_Active_Accts+Scaled_Primary_Overdue_Accts+
             New_Scaled_Primary_Sanct_Amt+New_Scaled_Primary_Disbursed_Amt+Scaled_Secondary_Num_Accts+
             Scaled_Secondary_Active_Accts+Scaled_Secondary_Overdue_Accts+
             New_Scaled_Secondary_Sanct_Amt+New_Scaled_Secondary_Disbursed_Amt+New_Scaled_Primary_Installment_Amount+
             New_Scaled_Secondary_Installment_Amount+Age+Employment.Type+State_ID+MobileNo_Avl_Flag+Aadhar_flag+
             PAN_flag+VoterID_flag+Driving_flag+Passport_flag+PERFORM_CNS.SCORE+Acct.Age.Num+Credit.History.Length.Num+
             NO.OF_INQUIRIES+NEW.ACCTS.IN.LAST.SIX.MONTHS+DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS,data=train,kernel=("radial"))


			 train$svmradialPreds <- predict(SVcL,train)
cm_SVcL <- table(train$svmradialPreds, train$loan_default)
Accuracy_svmRadial_train <- sum(diag(cm_SVcL))/sum(cm_SVcL)
Accuracy_svmRadial_train #1 for Trial on 180320192311
sensitivity_svmRadial_train <- cm_SVcL[1,1] / (cm_SVcL[1,1] + cm_SVcL[2,1])
sensitivity_svmRadial_train #1 for Trial on 180320192311
Specificity_svmRadial_train <- cm_SVcL[2,2] / (cm_SVcL[2,2] + cm_SVcL[1,2])
Specificity_svmRadial_train #1 for Trial on 180320192311

#Juxtapose on Test

test$Predict_SVC_Loan_Default<-predict(SVcL,test)
write.csv(test, file ="Final_Test_Preds.csv")