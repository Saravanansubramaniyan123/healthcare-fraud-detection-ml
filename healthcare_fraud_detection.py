import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import sklearn

import warnings
warnings.filterwarnings("ignore")

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Model persistence
import joblib
import json



# Visualization setup
import matplotlib.pyplot as plt

# Reproducibility - CRITICAL in production
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Project constants
LABELS = ["Normal", "Fraud"]

# Environment info for debugging
print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")


#loading train dataset
train = pd.read_csv("Train-1542865627584.csv")
train_ben_data = pd.read_csv("Train_Beneficiarydata-1542865627584.csv")
train_inptn_data = pd.read_csv("Train_Inpatientdata-1542865627584.csv")
train_outptn_data = pd.read_csv("Train_Outpatientdata-1542865627584.csv")

#loading test data
test = pd.read_csv("Test-1542969243754.csv")
test_ben_data = pd.read_csv("Test_Beneficiarydata-1542969243754.csv")
test_inptn_data = pd.read_csv("Test_Inpatientdata-1542969243754.csv")
test_outptn_data = pd.read_csv("Test_Outpatientdata-1542969243754.csv")

#checking shape of the each dataset
print("train dataset shape:",train.shape)
print("train benificiary data shape:",train_ben_data.shape)
print("train inpatient data shape:",train_inptn_data.shape)
print("train outpatient data shape:",train_outptn_data.shape)

print("test dataset shape:",test.shape)
print("test benificiary data shape:",test_ben_data.shape)
print("test inpatient data shape:",test_inptn_data.shape)
print("test outpatient data shape:",test_outptn_data.shape)

#lets check train and test dataset
print("train data:\n")
print(train.head(5))
print("test data:\n")
print(test.head(5))

#lets check provider details is unique or not
print(train["Provider"].value_counts().head(5))
print(train["Provider"].value_counts().shape)

#checking a null values
print("No.of.null values in train data:",train.isnull().sum().sum())
print("No.of.null values in test data:",test.isnull().sum().sum())

#setting up to display all the columns
pd.set_option("display.max.columns",None)
print(train_ben_data.head(5))

#checking a datatypes of a columns in the beneficiary data
print("train_beneficiary dataset datatypes:\n",train_ben_data.dtypes)
print("--"*50)
print("test_beneficiary dataset datatypes:\n",test_ben_data.dtypes)

#checking a null values
print("Null values in train_ben_data:",train_ben_data.isnull().sum())
print("Null values in test_ben_data:",test_ben_data.isnull().sum())

"""all the disease contains 1->yes and 2->No so we have to convert 2 into 0."""
train_ben_data = train_ben_data.replace({"ChronicCond_Alzheimer":2,
"ChronicCond_Heartfailure":2,
"ChronicCond_KidneyDisease":2,
"ChronicCond_Cancer":2,
"ChronicCond_ObstrPulmonary":2,
"ChronicCond_Depression":2,
"ChronicCond_Diabetes":2,
"ChronicCond_IschemicHeart":2,
"ChronicCond_Osteoporasis":2,
"ChronicCond_rheumatoidarthritis":2,
"ChronicCond_stroke":2},0)

test_ben_data = test_ben_data.replace({"ChronicCond_Alzheimer":2,
"ChronicCond_Heartfailure":2,
"ChronicCond_KidneyDisease":2,
"ChronicCond_Cancer":2,
"ChronicCond_ObstrPulmonary":2,
"ChronicCond_Depression":2,
"ChronicCond_Diabetes":2,
"ChronicCond_IschemicHeart":2,
"ChronicCond_Osteoporasis":2,
"ChronicCond_rheumatoidarthritis":2,
"ChronicCond_stroke":2},0)

train_ben_data = train_ben_data.replace({"RenalDiseaseIndicator":"Y"},1)
test_ben_data = test_ben_data.replace({"RenalDiseaseIndicator":"Y"},1)
print(test_ben_data.head(5))

#converting DOB and DOD into datatimetype based on that calculate the age for the person.
train_ben_data["DOB"] = pd.to_datetime(train_ben_data["DOB"])
train_ben_data["DOD"] = pd.to_datetime(train_ben_data["DOD"],errors = "ignore")
test_ben_data["DOB"] = pd.to_datetime(test_ben_data["DOB"])
test_ben_data["DOD"] = pd.to_datetime(test_ben_data["DOD"],errors = "ignore")
print(train_ben_data.dtypes)
print(test_ben_data.dtypes)

#creating new column age and add it into a test and train_ben_dataset
train_ben_data["Age"] = round(((train_ben_data["DOD"] - train_ben_data["DOB"]).dt.days)/365)
test_ben_data["Age"] = round(((test_ben_data["DOD"] - test_ben_data["DOB"]).dt.days)/365)
print(train_ben_data.head(10))
print(test_ben_data.head(10))


"""we can't able to calculate age for a persons who has a dod,so because of the we take last dod that registered in
the dataset and based on that we calculate the age for other persons"""
last_death = (train_ben_data["DOD"].max())
train_ben_data["Age"] = train_ben_data["Age"].fillna(round(((last_death - train_ben_data["DOB"]).dt.days)/365))
test_ben_data["Age"] = test_ben_data["Age"].fillna(round(((last_death - test_ben_data["DOB"]).dt.days)/365))
print(train_ben_data.head(10))
print(test_ben_data.head(10))

#creating a new column that shows the patient dead or not.
# using boolean indexing we are going to acheive this,if the DOD is not null the person is not dead "0" else "1".
train_ben_data.loc[train_ben_data["DOD"].isnull(),"WhetherDead"] = 0
train_ben_data.loc[train_ben_data["DOD"].notnull(),"WhetherDead"] = 1
print(train_ben_data.head(2))
test_ben_data.loc[test_ben_data["DOD"].isnull(),"WhetherDead"] = 0
test_ben_data.loc[test_ben_data["DOD"].notnull(),"WhetherDead"] = 1
print(test_ben_data.head(2))
print(train_inptn_data.head(5))
print(test_inptn_data.head(5))

#checking for null values in inpatient dataset
print("train inpatient data null values:\n",train_inptn_data.isnull().sum())
print(train_inptn_data.shape)
print("---"*100)
print("test inpatient data null values:\n",test_inptn_data.isnull().sum())
print(test_inptn_data.shape)

#converting a admission date and discharge date column into datetime type
train_inptn_data["AdmissionDt"] = pd.to_datetime(train_inptn_data["AdmissionDt"])
train_inptn_data["DischargeDt"] = pd.to_datetime(train_inptn_data["DischargeDt"])
test_inptn_data["AdmissionDt"] = pd.to_datetime(test_inptn_data["AdmissionDt"])
test_inptn_data["DischargeDt"] = pd.to_datetime(test_inptn_data["DischargeDt"])
print(train_inptn_data.dtypes)
print(test_inptn_data.dtypes)

#creating a new column Admit_For_Days
#we are adding 1 day extra day for each because if the patient admit and discharge in same day it should be considered as one not "0".
train_inptn_data["Admit_For_Days"] = ((train_inptn_data["DischargeDt"] - train_inptn_data["AdmissionDt"]).dt.days) + 1
test_inptn_data["Admit_For_Days"] = ((test_inptn_data["DischargeDt"] - test_inptn_data["AdmissionDt"]).dt.days) + 1
print(train_inptn_data.head(2))
test_inptn_data.head(2)

#min() and max() days patient is admitted.
print("max days a patient admitted in train data",train_inptn_data["Admit_For_Days"].max())
print("min days a patient addmitted in train data",train_inptn_data["Admit_For_Days"].min())
print("max days a patient admitted in train data",test_inptn_data["Admit_For_Days"].max())
print("min days a patient addmitted in train data",test_inptn_data["Admit_For_Days"].min())
print(train_outptn_data.head(3))
print(test_outptn_data.head(3))

#finding how many null values in the train and test of outpatient.
print("train outpatient data null values:\n",train_outptn_data.isnull().sum())
print("---"*100)
print("test outpatient data null values:\n",test_outptn_data.isnull().sum())

print("train dataset shape:",train.shape)
print("train benificiary data shape:",train_ben_data.shape)
print("train inpatient data shape:",train_inptn_data.shape)
print("train outpatient data shape:",train_outptn_data.shape)

print("test dataset shape:",test.shape)
print("test benificiary data shape:",test_ben_data.shape)
print("test inpatient data shape:",test_inptn_data.shape)
print("test outpatient data shape:",test_outptn_data.shape)

# merging both inpatient and outpatient data to make it as a  single data
train_all_ptn_data = pd.merge(train_inptn_data,train_outptn_data,
                              on = ['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode'],how = "outer")
test_all_ptn_data = pd.merge(test_inptn_data,test_outptn_data,
                              on = ['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
       'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
       'ClmAdmitDiagnosisCode'],how = "outer")
print(train_all_ptn_data.shape)
print(test_all_ptn_data.shape)
print(train_all_ptn_data.head(10))

#merging beneciary details data to all patient data
train_all_ptn_ben_data = pd.merge(train_all_ptn_data,train_ben_data,on = ["BeneID"],how = "inner")
test_all_ptn_ben_data = pd.merge(test_all_ptn_data,test_ben_data,on = ["BeneID"],how = "inner")
print(train_all_ptn_ben_data.shape)
print(test_all_ptn_ben_data.shape)

#merge train dataset(tells which provider fraud) with all patient details with beneficiary details.
train_prov_with_patn_ben = pd.merge(train_all_ptn_ben_data,train,on = ["Provider"],how = "inner")
test_prov_with_patn_ben = pd.merge(test_all_ptn_ben_data,test,on = ["Provider"],how = "inner")
print("shape of the train provider details with all patients data:",train_prov_with_patn_ben.shape)
print("shape of the test provider details with all patients data:",test_prov_with_patn_ben.shape)

#lets check null value percentage in each column for an entire dataset
print("Null value percentage for train provider with patient details:")
print(train_prov_with_patn_ben.isnull().sum()*100/len(train_prov_with_patn_ben))
print("---"*100)
print("Null value percentage for test provider with patient details:")
print(test_prov_with_patn_ben.isnull().sum()*100/len(test_prov_with_patn_ben))

#checking a datatype of an both test and train provider with patient details dataset
print(train_prov_with_patn_ben.dtypes)
print("---"*100)
print(test_prov_with_patn_ben.dtypes)

#plotting potential fraud classes proportion in train with merged data
sns.set_style('white',rc={'figure.figsize':(12,8)})
classes_count = train_prov_with_patn_ben["PotentialFraud"].value_counts()
print("potential fraud distribution in percentage:",classes_count*100/len(train_prov_with_patn_ben["PotentialFraud"]))
classes_count.plot(kind = "bar",figsize = (10,6),rot = 0,color = ["blue","red"])
plt.title("potential fraud distribution in train with merged data")
plt.xlabel("PotentialFraud")
plt.ylabel("No.of.potential fraud per class")
plt.grid(False)
plt.show()

#plotting potentialfraud classes distribution in train data(only have providerid and he is fraud or not)
count_classes_provider = train["PotentialFraud"].value_counts()
print("percentage of classes distribution in train data alone:",count_classes_provider*100/len(train))
count_classes_provider.plot(kind = "bar",rot = 0,figsize = (10,6),color = ["blue","red"])#rot = 90 means rotate xticks in 90 degree that means it looks like vertical.
plt.ylabel("No.of.potential fraud per class")
plt.grid(False)
plt.show()

# frequency distribution among beneficiary statewise.
bene_count_statewise_per = (train_prov_with_patn_ben["State"].value_counts())*100/len(train_prov_with_patn_ben["State"])
print("statewise beneficiary id distribution:\n",bene_count_statewise_per)
#bar chart
bene_count_statewise_per.plot(kind = "bar",figsize = (16,12),rot = 0)
plt.yticks(range(0,10,2),("0%","2%","4%","6%","8%"))
plt.title("distribution of beneficiary id among states",fontsize = 12)
plt.xlabel("State ID")
plt.ylabel("Percentage of beneficiary %")
plt.show()

#reacewise beneficaiary distribution
count_race = (train_ben_data["Race"].value_counts())*100/len(train_ben_data)
print("Race wise beneficiary distribution:\n",count_race)
count_race.plot(kind = "bar",rot = 0,figsize = (10,6))
plt.title("Race-Wise beneficiary distribution",fontsize = 12)
plt.ylabel("Percentage of beneficiary distribution among each race")
plt.yticks(range(0,100,10),("0%","10%","20%","30%","40%","50%","60%","70%","80%","90%"))
plt.show()

print(train_prov_with_patn_ben["ClmProcedureCode_1"].value_counts().iloc[:10].index)

#top 10 procedures in clmprocedurecode_1 that involve in heathcare fraud
sns.countplot(x = "ClmProcedureCode_1",hue = "PotentialFraud",data = train_prov_with_patn_ben
             ,order=train_prov_with_patn_ben.ClmProcedureCode_1.value_counts().iloc[:10].index)#without order it considers null values and creates big plot so because of that we can't able to plot a bar.
plt.title("Top 10 clmprocedurecode_1 invoved in healthcare fraud")
plt.show()

#top 10 count of ClmDiagnosisCode_1 involed in healthcare fraud.
sns.countplot(x = "ClmDiagnosisCode_1",hue = "PotentialFraud",data = train_prov_with_patn_ben
              ,order = train_prov_with_patn_ben.ClmDiagnosisCode_1.value_counts().iloc[0:10].index)
plt.title("Top 10 ClmDiagnosisCode_1 count involved in healthcare fraud")
plt.show()

#top-10 AttendingPhysician  involved in halthcarefraud
plt.figure(figsize = (16,12))
sns.countplot(x = "AttendingPhysician",hue = "PotentialFraud",data = train_prov_with_patn_ben
              ,order = train_prov_with_patn_ben["AttendingPhysician"].value_counts().iloc[:20].index)
plt.title("Top 10 attending physicians involved in fraud")
plt.xticks(rotation=90)
plt.show()

print(train_prov_with_patn_ben.dtypes)

#IPAnnualReimbursementAmt vs IPAnnualDeductibleAmt
sns.lmplot(x = "IPAnnualDeductibleAmt",y = "IPAnnualReimbursementAmt",data = train_prov_with_patn_ben
           ,hue = "PotentialFraud",col = "PotentialFraud",fit_reg = False)
plt.show()
"""There is no visible difference in the graph"""

#DeductibleAmtPaid vs InscClaimAmtReimbursed in both fraud and non fraud
sns.lmplot(y = "InscClaimAmtReimbursed",x = "DeductibleAmtPaid",data = train_prov_with_patn_ben
           ,hue = "PotentialFraud",col = "PotentialFraud",fit_reg = False)
plt.show()
"""DeductibleAmtPaid vs InscClaimAmtReimbursed for both fraud and non fraud looks like very same
we can't able to differentiate"""

#lets check age vs InscClaimAmtReimbursed in both fraud and non fraud
plt.figure(figsize = (16,12))
plt.subplot(2,1,1)
x = train_prov_with_patn_ben[train_prov_with_patn_ben.PotentialFraud == "Yes"].Age
y = train_prov_with_patn_ben[train_prov_with_patn_ben.PotentialFraud == "Yes"].InscClaimAmtReimbursed
plt.scatter(x,y)
plt.title("Fraud")
plt.ylabel("Insurance Claim Amout Reimbursed")
plt.xlabel('Age (in Years)')

plt.subplot(2,1,2)
x = train_prov_with_patn_ben[train_prov_with_patn_ben.PotentialFraud == "Yes"].Age
y = train_prov_with_patn_ben[train_prov_with_patn_ben.PotentialFraud == "Yes"].InscClaimAmtReimbursed
plt.title("Non - Fraud")
plt.ylabel("Insurance Claim Amout Reimbursed")
plt.xlabel('Age (in Years)')
plt.scatter(x,y)

plt.suptitle("Age vs Insurance Claim Amount Reimbursed (Fraud vs Non-Fraud)",fontsize = 16,y=0.95)
plt.show()

#Before appending the train data to the test data,we are creating a copy of test_data for future use.
test_prov_with_patn_ben_copy = test_prov_with_patn_ben
print("test_prov_with_patn_ben shape:",test_prov_with_patn_ben.shape)
print("test_prov_with_patn_ben_copy:",test_prov_with_patn_ben_copy.shape)
#As the test data doesn't have a target columns because of that we are only taking the columns that exists in the train data.
cols_test = test_prov_with_patn_ben.columns
#now appending the train data to test data
test_prov_with_patn_ben = pd.concat([test_prov_with_patn_ben,train_prov_with_patn_ben[cols_test]])
print("After combining both train and test expected no of collumns in test data:",(test_prov_with_patn_ben_copy.shape[0]) + (train_prov_with_patn_ben.shape[0]))
print("Total no of columns in test data :",test_prov_with_patn_ben.shape)

# we group the entire dataset based on Provider.
# columns that we are going to calculate a average.
avg_cols = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt',
            'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt',
            'Age', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'Admit_For_Days']

# Using for loop we are calculating a average for each category in the column
for col in avg_cols:
    train_prov_with_patn_ben[f"PerProviderAvg_{col}"] = train_prov_with_patn_ben.groupby('Provider')[col].transform(
        'mean')

for col in avg_cols:
    test_prov_with_patn_ben[f"PerProviderAvg_{col}"] = test_prov_with_patn_ben.groupby('Provider')[col].transform(
        'mean')

# As we know we are created a new column 10 columns that are the averages of the aldready existing columns.
# so the new columns are getting added at the end of the dataset.
print("Test:", test_prov_with_patn_ben.shape)
print(test_prov_with_patn_ben.iloc[:, -10:].head(2))
print("Train:", train_prov_with_patn_ben.shape)
print(train_prov_with_patn_ben.iloc[:, -10:].head(2))

#we group the entire dataset based on BeneID.
# columns that we are going to calculate a average.
avg_cols = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt',
                'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt',
                'Admit_For_Days']

# Create all per-beneficiary features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerBeneIDAvg_{col}"] = train_prov_with_patn_ben.groupby('BeneID')[col].transform('mean')

# Create all per-beneficiary features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerBeneIDAvg_{col}"] = test_prov_with_patn_ben.groupby('BeneID')[col].transform('mean')

# we group the entire dataset based on other physician

# Create all per-other-physician features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerOtherPhysicianAvg_{col}"] = train_prov_with_patn_ben.groupby('OtherPhysician')[col].transform('mean')

# Create all per-other-physician features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerOtherPhysicianAvg_{col}"] = test_prov_with_patn_ben.groupby('OtherPhysician')[col].transform('mean')

# dataset groped based on operating physician

# Create all per-operating-physician features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerOperatingPhysicianAvg_{col}"] = train_prov_with_patn_ben.groupby('OperatingPhysician')[col].transform('mean')

# Create all per-operating-physician features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerOperatingPhysicianAvg_{col}"] = test_prov_with_patn_ben.groupby('OperatingPhysician')[col].transform('mean')

#Grouping based on attending physician

# Create all per-attending-physician features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerAttendingPhysicianAvg_{col}"] = train_prov_with_patn_ben.groupby('AttendingPhysician')[col].transform('mean')

# Create all per-attending-physician features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerAttendingPhysicianAvg_{col}"] = test_prov_with_patn_ben.groupby('AttendingPhysician')[col].transform('mean')

#grouping based on diagnosis group code

# Create all per-diagnosis-group-code features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerDiagnosisGroupCodeAvg_{col}"] = train_prov_with_patn_ben.groupby('DiagnosisGroupCode')[col].transform('mean')

# Create all per-diagnosis-group-code features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerDiagnosisGroupCodeAvg_{col}"] = test_prov_with_patn_ben.groupby('DiagnosisGroupCode')[col].transform('mean')

# Verify diagnosis group code features were created
diagnosis_features = [col for col in train_prov_with_patn_ben.columns if col.startswith('PerDiagnosisGroupCodeAvg_')]
print(f"Created {len(diagnosis_features)} per-diagnosis-group-code average features")

#grouping based on admit diagnosis code

# Create all per-claim-admit-diagnosis-code features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerClmAdmitDiagnosisCodeAvg_{col}"] = train_prov_with_patn_ben.groupby('ClmAdmitDiagnosisCode')[col].transform('mean')

# Create all per-claim-admit-diagnosis-code features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerClmAdmitDiagnosisCodeAvg_{col}"] = test_prov_with_patn_ben.groupby('ClmAdmitDiagnosisCode')[col].transform('mean')

# Verify claim admit diagnosis code features were created
clm_admit_features = [col for col in train_prov_with_patn_ben.columns if col.startswith('PerClmAdmitDiagnosisCodeAvg_')]
print(f"Created {len(clm_admit_features)} per-claim-admit-diagnosis-code average features")

### grouped based on per claim procedure code 1

# Create all per-claim-procedure-code-1 features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerClmProcedureCode_1Avg_{col}"] = train_prov_with_patn_ben.groupby('ClmProcedureCode_1')[col].transform('mean')

# Create all per-claim-procedure-code-1 features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerClmProcedureCode_1Avg_{col}"] = test_prov_with_patn_ben.groupby('ClmProcedureCode_1')[col].transform('mean')

# Verify claim procedure code 1 features were created
procedure_1_features = [col for col in train_prov_with_patn_ben.columns if col.startswith('PerClmProcedureCode_1Avg_')]
print(f"Created {len(procedure_1_features)} per-claim-procedure-code-1 average features")

### grouped based on claim procedure code 2

# Create all per-claim-procedure-code-2 features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerClmProcedureCode_2Avg_{col}"] = train_prov_with_patn_ben.groupby('ClmProcedureCode_2')[col].transform('mean')

# Create all per-claim-procedure-code-2 features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerClmProcedureCode_2Avg_{col}"] = test_prov_with_patn_ben.groupby('ClmProcedureCode_2')[col].transform('mean')

# Verify claim procedure code 2 features were created
procedure_2_features = [col for col in train_prov_with_patn_ben.columns if col.startswith('PerClmProcedureCode_2Avg_')]
print(f"Created {len(procedure_2_features)} per-claim-procedure-code-2 average features")

### grouped based on claim procedure code 3

# Create all per-claim-procedure-code-3 features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerClmProcedureCode_3Avg_{col}"] = (
        train_prov_with_patn_ben.groupby('ClmProcedureCode_3')[col].transform('mean')
    )

# Create all per-claim-procedure-code-3 features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerClmProcedureCode_3Avg_{col}"] = (
        test_prov_with_patn_ben.groupby('ClmProcedureCode_3')[col].transform('mean')
    )

# Verify claim procedure code 3 features were created
procedure_3_features = [col for col in train_prov_with_patn_ben.columns
                        if col.startswith('PerClmProcedureCode_3Avg_')]
print(f"Created {len(procedure_3_features)} per-claim-procedure-code-3 average features")

### grouped based on claim diagnosis code 1

# Create all per-claim-diagnosis-code-1 features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerClmDiagnosisCode_1Avg_{col}"] = (
        train_prov_with_patn_ben.groupby('ClmDiagnosisCode_1')[col].transform('mean')
    )

# Create all per-claim-diagnosis-code-1 features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerClmDiagnosisCode_1Avg_{col}"] = (
        test_prov_with_patn_ben.groupby('ClmDiagnosisCode_1')[col].transform('mean')
    )

# Verify claim diagnosis code 1 features were created
diagnosis_1_features = [col for col in train_prov_with_patn_ben.columns
                        if col.startswith('PerClmDiagnosisCode_1Avg_')]
print(f"Created {len(diagnosis_1_features)} per-claim-diagnosis-code-1 average features")

### Average features grouped by ClmDiagnosisCode_2

# Create all per-claim-diagnosis-code-2 features for train
for col in avg_cols:
    train_prov_with_patn_ben[f"PerClmDiagnosisCode_2Avg_{col}"] = (
        train_prov_with_patn_ben.groupby('ClmDiagnosisCode_2')[col].transform('mean')
    )

# Create all per-claim-diagnosis-code-2 features for test
for col in avg_cols:
    test_prov_with_patn_ben[f"PerClmDiagnosisCode_2Avg_{col}"] = (
        test_prov_with_patn_ben.groupby('ClmDiagnosisCode_2')[col].transform('mean')
    )

# Verify claim diagnosis code 2 features were created
diagnosis_2_features = [col for col in train_prov_with_patn_ben.columns
                       if col.startswith('PerClmDiagnosisCode_2Avg_')]
print(f"Created {len(diagnosis_2_features)} per-claim-diagnosis-code-2 average features")

# Average features grouped by ClmDiagnosisCode_3

# Using predefined avg_cols list
for col in avg_cols:
    train_prov_with_patn_ben[f'ClmDiagnosisCode_3_{col}_avg'] = (
        train_prov_with_patn_ben.groupby('ClmDiagnosisCode_3')[col].transform('mean')
    )

for col in avg_cols:
    test_prov_with_patn_ben[f'ClmDiagnosisCode_3_{col}_avg'] = (
        test_prov_with_patn_ben.groupby('ClmDiagnosisCode_3')[col].transform('mean')
    )

# Verify claim diagnosis code 3 features were created
diagnosis_3_features = [
    col for col in train_prov_with_patn_ben.columns
    if col.startswith('ClmDiagnosisCode_3') and col.endswith('_avg')
]
print('Created', len(diagnosis_3_features), 'per-claim-diagnosis-code-3 average features')


# Average features grouped by ClmDiagnosisCode_4
# Using predefined avg_cols list
for col in avg_cols:
    train_prov_with_patn_ben[f'ClmDiagnosisCode_4_{col}_avg'] = (
        train_prov_with_patn_ben.groupby('ClmDiagnosisCode_4')[col].transform('mean')
    )

for col in avg_cols:
    test_prov_with_patn_ben[f'ClmDiagnosisCode_4_{col}_avg'] = (
        test_prov_with_patn_ben.groupby('ClmDiagnosisCode_4')[col].transform('mean')
    )

# Verify claim diagnosis code 4 features were created
diagnosis_4_features = [
    col for col in train_prov_with_patn_ben.columns
    if col.startswith('ClmDiagnosisCode_4') and col.endswith('_avg')
]
print('Created', len(diagnosis_4_features), 'per-claim-diagnosis-code-4 average features')

### Grouping based on different combinations and make count of them based on claim ID for both train and test data.

# 1) Define your partner cols & explicit combos (same as before)
two_way_feats = [
    'BeneID', 'AttendingPhysician', 'OtherPhysician', 'OperatingPhysician',
    'ClmAdmitDiagnosisCode',
    'ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3',
    'ClmProcedureCode_4','ClmProcedureCode_5',
    'ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3',
    'ClmDiagnosisCode_4','ClmDiagnosisCode_5','ClmDiagnosisCode_6',
    'ClmDiagnosisCode_7','ClmDiagnosisCode_8','ClmDiagnosisCode_9',
    'DiagnosisGroupCode'
]

three_way_feats = [
    ['BeneID','AttendingPhysician'],
    ['BeneID','OtherPhysician'],
    ['BeneID','OperatingPhysician'],
    ['BeneID','ClmProcedureCode_1'],
    ['BeneID','ClmDiagnosisCode_1']
]

four_way_feats = [
    ['BeneID','AttendingPhysician','ClmProcedureCode_1'],
    ['BeneID','AttendingPhysician','ClmDiagnosisCode_1'],
    ['BeneID','ClmDiagnosisCode_1','ClmProcedureCode_1']
]

# 2) Build your full list of group‐by key‐lists
groupings = []
groupings.append(['Provider'])                                   # 1‐way
groupings += [['Provider', c] for c in two_way_feats]             # 2‐way
groupings += [['Provider'] + combo for combo in three_way_feats]  # 3‐way
groupings += [['Provider'] + combo for combo in four_way_feats]   # 4‐way

# 3) Loop once to add all ClmCount_… features into both train & test
for grp in groupings:
    col_name = 'ClmCount_' + '_'.join(grp)
    train_prov_with_patn_ben[col_name] = (
        train_prov_with_patn_ben.groupby(grp)['ClaimID']
                               .transform('count')
    )
    test_prov_with_patn_ben[col_name] = (
        test_prov_with_patn_ben.groupby(grp)['ClaimID']
                               .transform('count')
    )

print(f" Created {len(groupings)} ClmCount_… features on train & test")

#checking the shape of both train and test data
print("train_prov_with_patn_ben shape :",train_prov_with_patn_ben.shape)
print("test_prov_with_patn_ben shape :",test_prov_with_patn_ben.shape)

#Lets check the unique values in the diagnosis code 1 column
# we are forcing converting the each values in the column into str type because if it is a Nan or othe values exits it throughs an error.
diagnosis_code1 = train_prov_with_patn_ben["ClmDiagnosisCode_1"].astype(str).str[0:2]
print(diagnosis_code1.unique())

#Filling a Numeric column null values with 0
col_num = train_prov_with_patn_ben.select_dtypes([np.number]).columns #np.number is a alias for all numerical datatypes.
train_prov_with_patn_ben[col_num] = train_prov_with_patn_ben[col_num].fillna(0)
test_prov_with_patn_ben[col_num] = test_prov_with_patn_ben[col_num].fillna(0)

#As we extracted all informations as numerical values now we are going to drop the original columns from the data.
#we do this because we extracted all the usefull informations from this original column that column is become useless now.
cols = train_prov_with_patn_ben.columns
print(cols[0:58])

remove_these_columns=['BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',
       'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
       'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
       'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
       'ClmAdmitDiagnosisCode', 'AdmissionDt',
       'DischargeDt', 'DiagnosisGroupCode','DOB', 'DOD',
        'State', 'County']

train_category_removed=train_prov_with_patn_ben.drop(axis=1,columns=remove_these_columns)
test_category_removed=test_prov_with_patn_ben.drop(axis=1,columns=remove_these_columns)
print("train shape:",train_category_removed.shape)
print("test shape:",test_category_removed.shape)
print("train null values:",train_category_removed.isnull().sum().sum())
print("test null values:",test_category_removed.isnull().sum().sum())

#for features race and gender we are going to done a one hot encoding.
#Type conversion
train_category_removed["Gender"] = train_category_removed["Gender"].astype("category")
test_category_removed["Gender"] = test_category_removed["Gender"].astype("category")
train_category_removed["Race"] = train_category_removed["Race"].astype("category")
test_category_removed["Race"] = test_category_removed["Race"].astype("category")

#Dummification
train_category_removed = pd.get_dummies(train_category_removed,columns = ["Gender","Race"],drop_first = True)
test_category_removed = pd.get_dummies(test_category_removed,columns = ["Gender","Race"],drop_first = True)

#checking the columns
print(train_category_removed.head().T) #T denotes transpose

#checking that both train and test are merged
print(test_category_removed.iloc[135391:135393])

#using map function
train_category_removed["PotentialFraud"] = train_category_removed["PotentialFraud"].map({"Yes":1,"No":0}).astype('int64')
print(train_category_removed.dtypes)
print("train_category_removed potential fraud min value:",train_category_removed["PotentialFraud"].min())
print("train_category_removed potential fraud min value:",train_category_removed["PotentialFraud"].max())
print(train_category_removed.head())

print("shape of test data before removing appended train data :",test_category_removed.shape)
#updating test data by removing appended train data.
test_category_removed = test_category_removed.iloc[:135392]
print("shape of test data after removed appended train data :",test_category_removed.shape)
#checking the last values to confirm that the appended train data has been removed.
print(test_category_removed.head())

#to display all the rows
pd.set_option("display.max.rows",None)
print(train_category_removed.dtypes)

#lets aggregate based on the provider and potentialfraud for train data & provider for test.
#it includes all features except reneldiseaseindicator because it contains string value.
train_category_removed_groupedby_prov = train_category_removed.groupby(["Provider","PotentialFraud"],as_index = False).sum(numeric_only = True)
test_category_removed_groupedby_prov = test_category_removed.groupby(["Provider"],as_index = False).sum(numeric_only = True)
print(test_category_removed_groupedby_prov.head(2))
print("Providers in train:",train_category_removed_groupedby_prov.shape)
print("Providers in test:",test_category_removed_groupedby_prov.shape)

#seperate x & y data
x = train_category_removed_groupedby_prov.drop(["Provider","PotentialFraud"],axis = 1)
y = train_category_removed_groupedby_prov["PotentialFraud"]

#standardizing a "x" data.
sc = StandardScaler()
sc.fit(x)
x_std = sc.transform(x)

x_test_std = sc.transform(test_category_removed_groupedby_prov.iloc[:,1:])
print(x_test_std[:3,:])
print(x_std[:3,:])
print("X shape :",x_std.shape)

#train and validation split
x_train,x_val,y_train,y_val = train_test_split(x_std,y,test_size = 0.3,random_state = 42,stratify = y,shuffle = True)
print("x train shape:",x_train.shape)
print("x val shape:",x_val.shape)
print("x train shape:",y_train.shape)
print("y val shape:",y_val.shape)

#Model Building
#logisticregressioncv for choosing a best regularization parameter c.
from sklearn.linear_model import LogisticRegressionCV
log_mod = LogisticRegressionCV(cv = 10,class_weight = "balanced",random_state = 42)
"""class_weight = "balanced" is for giving balanced weightage for all the clases even if the one class is count wise
very lower than the another class."""
log_mod.fit(x_train,y_train)

#Lets predict the probability of 0 and 1 for both x_train and y_train.To check the model is overfitted or underfitted.
x_train_pred_prob = log_mod.predict_proba(x_train)
x_val_pred_prob = log_mod.predict_proba(x_val)
print(x_train_pred_prob[:3])
print(x_val_pred_prob[:3])

#lets check the model is overfitted or underfitted using distplot
plt.figure(figsize = (12,8))
sns.distplot(x_train_pred_prob[:,1])#we are only taking a 1's probability.
sns.distplot(x_val_pred_prob[:,1])
plt.title("Probability of 1 become prediction in both x_train and y_val")
plt.xlim([0,1])
plt.tight_layout()
plt.show()

#roc curve
from sklearn.metrics import roc_curve,auc,precision_recall_curve
#roc_curve tests a different threshold and returns threshold value and their respective fpr and tpr.
fpr, tpr, thresholds = roc_curve(y_val,x_val_pred_prob[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr,tpr,label = f'ROC Curve (AUC = {roc_auc:.2f}%)',lw = 1)
plt.plot([0,1],[0,1],ls = "--")

for label in range(1,10,1):
    x = (10 - label)/10
    y = (10 - label)/10
    plt.text(x,y,thresholds[label*15],fontdict={'size': 14})

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.01])
plt.legend()
plt.show()

#precision vs recall
#here we are actually considering a fraud(1) positives only
precision,recall,threshold = precision_recall_curve(y_val,x_val_pred_prob[:,1])
plt.plot(precision,recall)
plt.xlabel("precision")
plt.ylabel("recall")
plt.title("precision vs recall")
plt.show()

#tpr vs fpr
sns.distplot(tpr)
sns.distplot(fpr)
plt.title("tpr vs fpr")
plt.xlabel("Probability")
plt.ylabel("Density")
plt.show()

print(y_train[:5])

#based on the threshold we find the fraud or not fraud
log_train_pred = (x_train_pred_prob[:,1] > 0.60)
log_val_pred = (x_val_pred_prob[:,1] > 0.60)

#model evaluation
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,roc_auc_score

cm_train = confusion_matrix(y_train,log_train_pred,labels = [1,0])
cm_val = confusion_matrix(y_val,log_val_pred,labels = [1,0])
print("x train data confusion matrix:\n",cm_train)
print("x validation data confusion matrix:\n",cm_val)

accuracy_x_train = accuracy_score(y_train,log_train_pred)
accuracy_x_val = accuracy_score(y_val,log_val_pred)
print("x train data accuracy score:",accuracy_x_train)
print("x val data accuracy score:",accuracy_x_val)

prec_x_train = precision_score(y_train,log_train_pred)
prec_x_val = precision_score(y_val,log_val_pred)
print("x train data precision score:",prec_x_train)
print("x val data precision score:",prec_x_val)

recall_x_train = recall_score(y_train,log_train_pred)
recall_x_val = recall_score(y_val,log_val_pred)
print("x train data recall score:",recall_x_train)
print("x val data recall score:",recall_x_val)

f1_x_train = f1_score(y_train,log_train_pred)
f1_x_val = f1_score(y_val,log_val_pred)
print("x train data f1 score:",f1_x_train)
print("x val data f1 score:",f1_x_val)

print("area under curve for x_train data:",roc_auc_score(y_train,log_train_pred))
print("area under curve for x_val data:",roc_auc_score(y_val,log_val_pred))

# Save Logistic Regression model
joblib.dump(log_mod, 'logistic_regression_threshold_60.joblib')
joblib.dump(sc, 'logistic_regression_scaler.joblib')

# Save performance metrics
log_metrics = {
    'model_type': 'LogisticRegression',
    'threshold': 0.60,
    'train_accuracy': accuracy_x_train,
    'val_accuracy': accuracy_x_val,
    'train_f1': f1_x_train,
    'val_f1': f1_x_val
}

with open('logistic_regression_metadata.json', 'w') as f:
    json.dump(log_metrics, f, indent=2)

# Test loading
loaded_log_model = joblib.load('logistic_regression_threshold_60.joblib')
loaded_log_scaler = joblib.load('logistic_regression_scaler.joblib')
print("Logistic Regression saved successfully!")

#test data prediction
log_test_pred = (log_mod.predict_proba(x_test_std)[:,1] > 0.60)
"""we convert it into a dataframe and replace the 1->yes and 0->no,then combine provider id with the predicted fraud or not"""
log_test_pred = pd.DataFrame(log_test_pred)
print(log_test_pred.head())
#replacing the value 1 with "Yes" and 0 with "No"
replacement = {1:"Yes",0:"No"}
labels = log_test_pred[0].apply(lambda x:replacement[x])
print(labels.value_counts())
labels.head()

#now we are combining a test_category_removed_groupedby_prov provider column and fraud or not prediction that we found above that is labels.
#And then save it as a submission log file.The file contains provider id and they are fraud or not
submission_log = pd.DataFrame({"Provider":test_category_removed_groupedby_prov.Provider})
submission_log["Potential_fraud"] = labels
print(submission_log.shape)
submission_log.head()

#submission file
submission_log.to_csv("Submission_logistic_regresssion_threshold_60.csv",index = False)


#lets try random forest to do the same
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 500,class_weight = "balanced",random_state = 42,max_depth = 4)
rfc.fit(x_train,y_train)

#roc curve
x_val_pred_prob = rfc.predict_proba(x_val)
fpr,tpr,threshold = roc_curve(y_val,x_val_pred_prob[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr,lw = 1,label = f"roc curve (AUC = {roc_auc:.2f}%)")
plt.plot([0,1],[0,1],ls = "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.01])
plt.legend()
plt.show()

#distribution plot for both fpr and tpr
sns.distplot(tpr,color = "darkblue")
sns.distplot(fpr,color = "red")
plt.text(0.1,5,"Negatives",color = "red")
plt.text(0.8,5,"Positives",color = "darkblue")
plt.xlabel("Probability")
plt.xlim([-0.2,1.2])
plt.ylabel("Distribution")
plt.show()

#by default threshold is 0.5 so we don't need to choose a thresholt
rfc_train_pred = rfc.predict(x_train)
rfc_val_pred = rfc.predict(x_val)

#model evaluation
rfc_cm_train = confusion_matrix(y_train,rfc_train_pred,labels = [1,0])
rfc_cm_val = confusion_matrix(y_val,rfc_val_pred,labels = [1,0])
print("Random forest train data prediction confusion matrix:\n",rfc_cm_train)
print("Random forest val data prediction confusion matrix:\n",rfc_cm_val)

print("Random forest train data prediction accuracy score:",accuracy_score(y_train,rfc_train_pred))
print("Random forest val data prediction accuracy score:",accuracy_score(y_val,rfc_val_pred))

print("Random forest train data prediction precision score:",accuracy_score(y_train,rfc_train_pred))
print("Random forest val data prediction precision score:",accuracy_score(y_val,rfc_val_pred))

print("Random forest train data prediction recall score:",recall_score(y_train,rfc_train_pred))
print("Random forest val data prediction recall score:",recall_score(y_val,rfc_val_pred))

print("Random forest train data prediction f1 score:",f1_score(y_train,rfc_train_pred))
print("Random forest val data prediction f1 score:",f1_score(y_val,rfc_val_pred))

print("Random forest train data prediction area under curve score:",roc_auc_score(y_train,rfc_train_pred))
print("Random forest val data prediction area under curve score:",roc_auc_score(y_val,rfc_val_pred))

#RandomForestFeature importance
feature_list = list(test_category_removed_groupedby_prov.columns[1:])#here the providerid column also included so we have to remove it because we removed that column in the dataset given to the model.
print(feature_list[:5])
importances = list(rfc.feature_importances_)#this returns feature importance scores in the order that we given the data to the model.
print(importances[:5])
#using zip function we merge the feature name and their importances
feature_importances = [(feature,round(score,2))for feature,score in zip(feature_list,importances)]
#creating a dataframe
df_feature_importances = pd.DataFrame(feature_importances,columns = ["Features","Scores"])
df_feature_importances.set_index("Features",inplace = True)
#now sorting the dataframe based on the scores of the feature
df_feature_importances = df_feature_importances.sort_values(by = "Scores",ascending = False)
print(df_feature_importances.head())

# Save Random Forest model without
joblib.dump(rfc, 'random_forest.joblib')
joblib.dump(sc, 'random_forest_scaler.joblib')  # Same scaler as logistic regression

# Save performance metrics
rf_metrics = {
    'model_type': 'RandomForest',
    'n_estimators': 500,
    'max_depth': 4,
    'train_accuracy': accuracy_score(y_train, rfc_train_pred),
    'val_accuracy': accuracy_score(y_val, rfc_val_pred),
    'train_f1': f1_score(y_train, rfc_train_pred),
    'val_f1': f1_score(y_val, rfc_val_pred)
}

with open('random_forest_metadata.json', 'w') as f:
    json.dump(rf_metrics, f, indent=2)

# Test loading
loaded_rf_model = joblib.load('random_forest.joblib')
loaded_rf_scaler = joblib.load('random_forest_scaler.joblib')
print("Random Forest saved successfully")

#prediction of test data
rfc_test_prediction = rfc.predict(x_test_std)
print(rfc_test_prediction[:5])
#merging the predicted values with their respective providerid
merged_prov_pf = pd.DataFrame({
    "Provider":test_category_removed_groupedby_prov.Provider,
    "PotentialFraud":rfc_test_prediction})
#replace 1 and 0 with yes and no
merged_prov_pf["PotentialFraud"] = merged_prov_pf["PotentialFraud"].replace({0:"No",1:"Yes"})
print(merged_prov_pf.head())

#submission
print(merged_prov_pf.shape)
merged_prov_pf.to_csv("Submission_Random_Forest_Classifier.csv",index = False)

#pca
print(train_category_removed_groupedby_prov.head(2))
print(test_category_removed_groupedby_prov.head(2))

#Standardizing a  both train and val data
std = StandardScaler()
std.fit(train_category_removed_groupedby_prov.iloc[:,2:])#for train data we are removing provider and potentialfraud columns.
train_category_removed_groupedby_prov_scaled = std.transform(train_category_removed_groupedby_prov.iloc[:,2:])
test_category_removed_groupedby_prov_scaled = std.transform(test_category_removed_groupedby_prov.iloc[:,1:])#here we removed a provider column because id doesn't give any value to model.
#converting both the scaled into dataframe
train_category_removed_groupedby_prov_scaled = pd.DataFrame(train_category_removed_groupedby_prov_scaled)
test_category_removed_groupedby_prov_scaled = pd.DataFrame(test_category_removed_groupedby_prov_scaled)
print(train_category_removed_groupedby_prov_scaled.shape)
print(test_category_removed_groupedby_prov_scaled.shape)

#PCA Maximum Variance
from sklearn.decomposition import PCA
pca = PCA(n_components = 29)
pca.fit(train_category_removed_groupedby_prov_scaled)

#pca.explained_variance_ratio_ returns the variance of each fitted data.
#np.round is just like a round()
print("pca explained variance :\n",np.round(pca.explained_variance_ratio_,3))
#now transform both train and test data and then the total number of datapoints decomposeto the given datapoints
train_pca = pca.transform(train_category_removed_groupedby_prov_scaled)
test_pca = pca.transform(test_category_removed_groupedby_prov_scaled)
print("train pca shape:",train_pca.shape)
print("test pca shape:",test_pca.shape)

#now converting both train and test data into dataframe
train_pca = pd.DataFrame(train_pca)
test_pca = pd.DataFrame(test_pca)
print("train pca dataframe:",train_pca.head())
print("test pca dataframe:",test_pca.head())
#Now adding a potential fraud column that we removed before a train a data now we are adding that again into a pca.
train_pca["PotentialFraud"] = train_category_removed_groupedby_prov.PotentialFraud
train_pca.to_csv("train_pca.csv",index = False)
test_pca.to_csv("test_pca.csv",index = False)
print(train_pca.head())

#isolation forest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Split out features and label from your PCA DataFrame
X_pca_train = train_pca.drop(columns="PotentialFraud")
y_pca_train = train_pca["PotentialFraud"].astype(int)   # target

# Fit Isolation Forest (unsupervised!)
iso = IsolationForest(
    n_estimators    = 100,
    contamination   = 0.09,
    random_state    = 42
)
iso.fit(X_pca_train)

# Predict on TRAIN to see how well it isolates known fraud
# -1 → anomaly (fraud), +1 → normal
raw_train_pred = iso.predict(X_pca_train)
y_train_if = (raw_train_pred == -1).astype(int)

print("=== Train Classification Report ===\n")
print(classification_report(y_pca_train, y_train_if,
      target_names=["Normal","Fraud"]))

cm = confusion_matrix(y_pca_train, y_train_if, labels=[0,1])
print("=== Train Confusion Matrix ===")
print(pd.DataFrame(cm,
                   index=["TrueNormal","TrueFraud"],
                   columns=["PredNormal","PredFraud"]))

# Save Isolation Forest model using joblib
joblib.dump(iso, 'isolation_forest.joblib')
joblib.dump(std, 'isolation_forest_pca_scaler.joblib')  # PCA scaler
joblib.dump(pca, 'isolation_forest_pca_transformer.joblib')  # PCA transformer

# Saving a metrics as a json file
iso_metrics = {
    'model_type': 'IsolationForest',
    'contamination': 0.09,
    'n_estimators': 100,
    'pca_components': 29
}

with open('isolation_forest_metadata.json', 'w') as f:
    json.dump(iso_metrics, f, indent=2)

# loading a svaed model
loaded_iso_model = joblib.load('isolation_forest.joblib')
loaded_iso_scaler = joblib.load('isolation_forest_pca_scaler.joblib')
loaded_pca = joblib.load('isolation_forest_pca_transformer.joblib')
print("Isolation Forest saved successfully")

# Now predict the test data and create a submission file
X_pca_test = test_pca.copy()   # your 1353×29 PCA DataFrame
raw_test_pred = iso.predict(X_pca_test)
y_test_if = (raw_test_pred == -1).astype(int)

#
submission = pd.DataFrame({
    "Provider"       : test_category_removed_groupedby_prov["Provider"],
    "PotentialFraud" : np.where(y_test_if==1, "Yes", "No")
})
print(submission.shape)
submission.to_csv("Submission_Isolation_Forest.csv",index = False)
print("\nSubmission value counts:")
print(submission["PotentialFraud"].value_counts())
print(submission.head())




