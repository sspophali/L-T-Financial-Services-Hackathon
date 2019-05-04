# -*- coding: utf-8 -*-
"""
Project : LTFS Hackthon

"""
# Loading Basic Libraries 
import pandas as pd
import numpy as np
import seaborn as sns

# Loading Classifiers 
from sklearn.metrics import classification_report
from sklearn.ensemble.forest import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Loading spliting libraries
from sklearn.model_selection import train_test_split
from sklearn import cross_validation

# Loading Evaluation Metrics 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# setting Graph parameters
sns.set(rc={'figure.figsize':(11,8)})
sns.set(style="whitegrid")
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))

# Function for Converting "period" columns to total months
def change_col_month(col):
    year = int(col.split()[0].replace('yrs',''))
    month = int(col.split()[1].replace('mon',''))
    return year*12+month

# Function for Calculating age out of DOB
def calculate_age(dob):
    if int(dob[6:]) < 19:
        year = int('20' + dob[6:])
    else:
        year = int('19' + dob[6:])    
    return (pd.to_datetime('today').year - year)

# Function for assigning bins based on Bureau Score
def assign_bin(val,bins):
    flag = 0
    for i in range(len(bins)-1):
        if val>=bins[i] and val<bins[i+1]:
            flag = 1
            if i == 0:
                return 'HIGH_RISK'
            elif i == 1:
                return 'MEDIUM_RISK'
            elif i == 2:
                return 'LOW_RISK'
            elif i == 3:
                return 'VERY_LOW_RISK'
            
# Function for printing Bins
def print_bin(bins):
    for i in range(len(bins)-1):
        print('Bin ['+str(i)+'] : '+str(bins[i])+' to '+str(bins[i+1]))


def data_info(loan_data):
    
    print("Data Snippet: ")
    print(loan_data.head())
    print("--------------------------------------------------------------------")
    
    print("Stats about Data")
    print(loan_data.info())
    print("--------------------------------------------------------------------")
    
    categorical = [var for var in loan_data.columns if loan_data[var].dtypes == 'O']
    print("Categolrical Varibales: ",len(categorical))
    for c in categorical:
        print(c)
    print("--------------------------------------------------------------------")
    numerical = [var for var in loan_data.columns if loan_data[var].dtypes != 'O']
    print("Numerical Varibales: ",len(numerical))
    for n in numerical:
        print(n)
    print("--------------------------------------------------------------------")
    print('Unique Values per column')    
    unique_values = loan_data.apply(lambda x: len(pd.unique(x)),axis = 0).sort_values(ascending = False) #unique rate and sort
    print(unique_values)
   
def prepare_data(loan_data):
    
    loan_data  = loan_data[['UniqueID','loan_default','disbursed_amount','asset_cost',
                        'ltv','branch_id','supplier_id','manufacturer_id','Date.of.Birth',
                        'Employment.Type','DisbursalDate','Driving_flag','PERFORM_CNS.SCORE',
                        'PERFORM_CNS.SCORE.DESCRIPTION','PRI.NO.OF.ACCTS','PRI.ACTIVE.ACCTS',
                        'PRI.OVERDUE.ACCTS','PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT',
                        'PRI.DISBURSED.AMOUNT','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.OVERDUE.ACCTS',
                        'SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT',
                        'PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT','NEW.ACCTS.IN.LAST.SIX.MONTHS',
                        'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH','NO.OF_INQUIRIES']]
    
    loan_data['Employment.Type'] = loan_data['Employment.Type'].fillna('Self employed')
    
    loan_data['age'] = loan_data['Date.of.Birth'].apply(calculate_age)
    #loan_data['age'].value_counts().plot(kind='bar')
    del loan_data['Date.of.Birth']
    loan_data['AVERAGE.ACCT.AGE'] = loan_data['AVERAGE.ACCT.AGE'].apply(change_col_month)
    loan_data['CREDIT.HISTORY.LENGTH'] = loan_data['CREDIT.HISTORY.LENGTH'].apply(change_col_month) 
    #print("Creating Bins for Bureau Score: ")
    bins=range(1,1002,200)
    #for b in bins:
    #   print(b)
    loan_data['Bureau_bin']= loan_data['PERFORM_CNS.SCORE'].apply(lambda x: assign_bin(x,bins))
    #print_bin(bins)
    #loan_data['Bureau_bin'].value_counts().plot(kind='bar')
    # we can create 3 categories which are having No credit Score
    # 'HIGH_RISK_New.Credit' - when age is less than 25 & ltv > 85 & self employed
    # 'MEDIUM_RISK_New.Credit' - when age is between 26 t0 50 & ltv > 85
    # 'LOW_RISK_New.Credit' - when age is more than 50 
    # lets create these bins in 'Bureau_bins'
    
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data.age<25) & (loan_data.ltv >= 85) & (loan_data['Employment.Type']=='Self employed'),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='HIGH_RISK_New.Credit'
                      
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data.age>25) & (loan_data.age<=50) & (loan_data.ltv >= 85),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='MEDIUM_RISK_New.Credit'
                                        
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data.age>50),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='LOW_RISK_New.Credit'

    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data['Bureau_bin'].isnull()) & (loan_data['Employment.Type'] == 'Self Employed'),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='MEDIUM_RISK_New.Credit'
    
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data['Bureau_bin'].isnull()) & (loan_data['Employment.Type'] == 'Salaried'),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='LOW_RISK_New.Credit'
    
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data['Bureau_bin'].isnull()) & (loan_data['manufacturer_id'] == 86),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='MEDIUM_RISK_New.Credit'

    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data['Bureau_bin'].isnull()) & (loan_data['manufacturer_id'] != 86),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='LOW_RISK_New.Credit'
    
    loan_data['Bureau_bin'].loc[loan_data['Bureau_bin'].isnull()]='VERY_LOW_RISK'
    
    return loan_data
    
def prepare_test_data(loan_data):
    
    loan_data  = loan_data[['UniqueID','disbursed_amount','asset_cost',
                        'ltv','branch_id','supplier_id','manufacturer_id','Date.of.Birth',
                        'Employment.Type','DisbursalDate','Driving_flag','PERFORM_CNS.SCORE',
                        'PERFORM_CNS.SCORE.DESCRIPTION','PRI.NO.OF.ACCTS','PRI.ACTIVE.ACCTS',
                        'PRI.OVERDUE.ACCTS','PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT',
                        'PRI.DISBURSED.AMOUNT','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.OVERDUE.ACCTS',
                        'SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT',
                        'PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT','NEW.ACCTS.IN.LAST.SIX.MONTHS',
                        'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH','NO.OF_INQUIRIES']]
    
    loan_data['Employment.Type'] = loan_data['Employment.Type'].fillna('Self employed')
    
    loan_data['age'] = loan_data['Date.of.Birth'].apply(calculate_age)
    #loan_data['age'].value_counts().plot(kind='bar')
    del loan_data['Date.of.Birth']
    loan_data['AVERAGE.ACCT.AGE'] = loan_data['AVERAGE.ACCT.AGE'].apply(change_col_month)
    loan_data['CREDIT.HISTORY.LENGTH'] = loan_data['CREDIT.HISTORY.LENGTH'].apply(change_col_month) 
    #print("Creating Bins for Bureau Score: ")
    bins=range(1,1002,200)
    #for b in bins:
    #   print(b)
    loan_data['Bureau_bin']= loan_data['PERFORM_CNS.SCORE'].apply(lambda x: assign_bin(x,bins))
    #print_bin(bins)
    #loan_data['Bureau_bin'].value_counts().plot(kind='bar')
    # we can create 3 categories which are having No credit Score
    # 'HIGH_RISK_New.Credit' - when age is less than 25 & ltv > 85 & self employed
    # 'MEDIUM_RISK_New.Credit' - when age is between 26 t0 50 & ltv > 85
    # 'LOW_RISK_New.Credit' - when age is more than 50 
    # lets create these bins in 'Bureau_bins'
    
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data.age<25) & (loan_data.ltv >= 85) & (loan_data['Employment.Type']=='Self employed'),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='HIGH_RISK_New.Credit'
                      
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data.age>25) & (loan_data.age<=50) & (loan_data.ltv >= 85),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='MEDIUM_RISK_New.Credit'
                                        
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data.age>50),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='LOW_RISK_New.Credit'

    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data['Bureau_bin'].isnull()) & (loan_data['Employment.Type'] == 'Self Employed'),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='MEDIUM_RISK_New.Credit'
    
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data['Bureau_bin'].isnull()) & (loan_data['Employment.Type'] == 'Salaried'),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='LOW_RISK_New.Credit'
    
    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data['Bureau_bin'].isnull()) & (loan_data['manufacturer_id'] == 86),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='MEDIUM_RISK_New.Credit'

    ind=loan_data.loc[(loan_data['PERFORM_CNS.SCORE']==0) & (loan_data['Bureau_bin'].isnull()) & (loan_data['manufacturer_id'] != 86),'Bureau_bin'].index
    loan_data['Bureau_bin'].loc[ind]='LOW_RISK_New.Credit'
    
    loan_data['Bureau_bin'].loc[loan_data['Bureau_bin'].isnull()]='VERY_LOW_RISK'
    
    return loan_data


def split_data(loan_data):
    
    # Here we are diving data into two sets
    # set 1 : data having Bureau score ZERO
    # set 2 : data having Bureau score NON-ZERO
    zero_data = loan_data[loan_data['PERFORM_CNS.SCORE']==0]
    nonzero_data = loan_data[loan_data['PERFORM_CNS.SCORE']!=0]
    
    # selecting features
    # in ZERO score we can delete all columns related to history of loanee
    zero_data = zero_data[['disbursed_amount','ltv','manufacturer_id','Employment.Type','Driving_flag','age','Bureau_bin','loan_default']]
    nonzero_data = nonzero_data[['ltv','Employment.Type','Driving_flag','PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
                             'age','Bureau_bin','disbursed_amount','loan_default','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH']]
    return zero_data, nonzero_data

def split_test_data(loan_data):
    
    # Here we are diving data into two sets
    # set 1 : data having Bureau score ZERO
    # set 2 : data having Bureau score NON-ZERO
    zero_data = loan_data[loan_data['PERFORM_CNS.SCORE']==0]
    nonzero_data = loan_data[loan_data['PERFORM_CNS.SCORE']!=0]
    
    # selecting features
    # in ZERO score we can delete all columns related to history of loanee
    zero_data = zero_data[['disbursed_amount','ltv','manufacturer_id','Employment.Type','Driving_flag','age','Bureau_bin']]
    nonzero_data = nonzero_data[['ltv','Employment.Type','Driving_flag','PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
                             'age','Bureau_bin','disbursed_amount','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH']]
    return zero_data, nonzero_data

def predict_evaluate_classifier(X_test,y_test,model):

    y_pred = model.predict(X_test)
    print("Classification Report: ")    
    print(classification_report(y_test,y_pred))
    print("--------------------------------------------------------------------")
    print("ROC Score: "+str(round(roc_auc_score(y_test,y_pred),2)))
    print("--------------------------------------------------------------------")
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------------------------------------------")
#    return y_pred
    

def train_model(data):

    dataset = pd.get_dummies(data,columns=['Employment.Type','Driving_flag','Bureau_bin'],drop_first=True)
    #dataset = pd.get_dummies(data,columns=['Employment.Type','Driving_flag'],drop_first=True)
    X = dataset.drop('loan_default',axis=1)
    y = dataset['loan_default']

    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,train_size=.8, stratify=y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
    
    rfc = RandomForestClassifier(class_weight='balanced',n_estimators=100)
    rfc.fit(X_train,y_train)
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_train,y_train)
    xgb = XGBClassifier(scale_pos_weight=3.4)
    xgb.fit(X_train,y_train)
    
    brfc = BalancedRandomForestClassifier(max_depth=4, random_state=0)
    brfc.fit(X_train,y_train)
    bbc = BalancedBaggingClassifier(n_estimators=100,random_state=42)
    bbc.fit(X_train,y_train)
    models = [rfc, lr, xgb, brfc, bbc]
    model_names = ['RandomForestClassifier','LogisticRegression','XGBClassifier','BalancedRandomForestClassifier','BalancedBaggingClassifier']    
    for m, n in zip(models,model_names):
        print('Classifier: '+n)
        predict_evaluate_classifier(X_test,y_test,m)
        
    return rfc, lr, xgb, brfc, bbc

'''def train_BBC(data):
    data = pd.get_dummies(data,columns=['Employment.Type','Driving_flag','Bureau_bin'],drop_first=True)
    X = data.drop('loan_default',axis=1)
    y = data['loan_default']
    brfc = BalancedRandomForestClassifier(max_depth=4, random_state=0)
    brfc.fit(X,y)
    return brfc
'''    
def test_BBC(data,model):
    data = pd.get_dummies(data,columns=['Employment.Type','Driving_flag','Bureau_bin'],drop_first=True)
    X = data.drop('loan_default',axis=1)
    y = data['loan_default']
    predictions = model.predict(X)
    print("Classification Report: ")    
    print(classification_report(y,predictions))
    print("--------------------------------------------------------------------")
    print("ROC Score: "+str(round(roc_auc_score(y,predictions),2)))
    print("--------------------------------------------------------------------")
    print("Confusion Matrix: ")
    print(confusion_matrix(y, predictions))
    print("--------------------------------------------------------------------")
    
    
'''def predict_evaluate_classifier(X_test,y_test,model):

    y_pred = model.predict(X_test)
    print("Classification Report: ")    
    print(classification_report(y_test,y_pred))
    print("--------------------------------------------------------------------")
    print("ROC Score: "+str(round(roc_auc_score(y_test,y_pred),2)))
    print("--------------------------------------------------------------------")
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------------------------------------------")
    return y_pred
'''    
# Loading Data
dataset_train = pd.read_csv("C://Sameer//Data Science//Aegis//Hackthon//LTFS//train_aox2Jxw//train.csv")
dataset_test = pd.read_csv("C://Sameer//Data Science//Aegis//Hackthon//LTFS//test.csv")

data_info(dataset_train)    

dataset_train = prepare_data(dataset_train)
dataset_test = prepare_test_data(dataset_test)

zero_data_train, nonzero_data_train = split_data(dataset_train)
zero_data_test, nonzero_data_test = split_test_data(dataset_test)

rfc, lr, xgb, brfc, bbc = train_model(zero_data_train)
rfc1, lr1, xgb1, brfc1, bbc1 = train_model(nonzero_data_train)

zero_data_test = pd.get_dummies(zero_data_test,columns=['Employment.Type','Driving_flag','Bureau_bin'],drop_first=True)
zero_predtiction = brfc.predict(zero_data_test)

nonzero_data_test = pd.get_dummies(nonzero_data_test,columns=['Employment.Type','Driving_flag','Bureau_bin'],drop_first=True)
nonzero_predtiction = xgb1.predict(nonzero_data_test)

loan_default = list(zero_predtiction)+list(nonzero_predtiction)
#len(loan_default)
UniqueID = list(dataset_test.loc[(dataset_test['PERFORM_CNS.SCORE']==0),'UniqueID'])+list(dataset_test.loc[(dataset_test['PERFORM_CNS.SCORE']!=0),'UniqueID'])
#len(UniqueID)

result = pd.DataFrame({
        'UniqueID':UniqueID,
        'loan_default':loan_default})

result.to_csv("C://Sameer//Data Science//Aegis//Hackthon//LTFS//result.csv",index=False)
        