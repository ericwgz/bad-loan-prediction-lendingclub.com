# Bad loan analysis using lendingclub public data (https://www.lendingclub.com/info/download-data.action)

# Written by Pin-Chih Su

# Tested in python 2.7.6, numpy 1.9.0, scipy-0.14.0, matplotlib.pyplot-1.3.1, sklearn 0.17.0 in Linux RedHat 5.0

import pandas as pd
#In order to display all the columns, bypass the pandas autodetect
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as pl
from sklearn.metrics import roc_curve, auc
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import svm
##from sklearn.neural_network import MLPClassifier # Not release yet
from sklearn.neural_network import BernoulliRBM

# Read the csv

df=pd.read_csv("LoanStats3c_clean.csv")

## Make the bad loans in the "loan_status" column as 0 and "good loans" as 1

##print pd.value_counts(df['loan_status'].values) # Descrptive stat of the 'loan_status' column

positive = ['Fully Paid']

negative = ['Charged Off']

# filter out any word that is not within positive & negative
##"Does not meet the credit policy. Status:1"
##"Does not meet the credit policy. Status:0" are filtered out above

filtered_df = df[df['loan_status'].isin(positive + negative)].copy()

filtered_df['loan_status'] = filtered_df['loan_status'].isin(positive).astype(int) #.astype(int) will print "True" as "1"

##print filtered_df['loan_status'].sum()
# Plot histogram of good and bad loans
##plt.xlabel("Good Loan = 1, Bad Loan =0")
##plt.ylabel("Count")
##plt.title("Good/Bad Loan Count")
##plt.hist(list(filtered_df['loan_status']))
##plt.savefig("2013-2014-broader-bad-loan-def.png")

# Make home owners in the "home_ownership" column as 1 and non-homeowner as 0

home_positive = ['OWN', 'MORTGAGE']

home_negative = ['RENT','NONE']

# filter out any word that is not within home_positive & home_negative

filtered_df = filtered_df[df['home_ownership'].isin(home_positive + home_negative)].copy()

filtered_df['home_ownership']=filtered_df['home_ownership'].isin(home_positive).astype(int)

# Make "verified" and "Source Verified" in the "verification_status" column as 1 and non-verified as 0

verification_positive = ['Verified', 'Source Verified']

verification_negative = ['Not Verified']

filtered_df = filtered_df[df['verification_status'].isin(verification_positive + verification_negative)].copy()

filtered_df['verification_status']=filtered_df['verification_status'].isin(verification_positive).astype(int)

# Make "36 months" in the "term" column as "1" and "60 months" as "0"

term36 = [' 36 months']

term60 = [' 60 months']

filtered_df = filtered_df[df['term'].isin(term36 + term60)].copy()

filtered_df['term']=filtered_df['term'].isin(term36).astype(int)

# Let us detect numeric only columns

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

newdf = filtered_df.select_dtypes(include=numerics)

# We decide to keep the following columns for machine learning from the result of another variable selection script

cols_to_keep = ['loan_status','dti','revol_bal','annual_inc','loan_amnt','total_acc']

ml_df= newdf[cols_to_keep]

##print train_cols => get column titles
#Output: Index([u'annual_inc', u'delinq_2yrs', u'home_ownership', u'inq_last_6mths',u'loan_amnt', u'pub_rec'],dtype='object')

train_cols = ml_df.columns[1:]

## Split training and test set using 7:3 ratio

X_train, X_test, y_train, y_test = train_test_split(ml_df[train_cols],ml_df['loan_status'],test_size=0.3,random_state=1)

### Logistic regression

logit = sm.Logit(y_train,X_train)

#fit the model
result = logit.fit()

print result.summary()

lr_pred = result.predict(X_test)

### Random Forest

rf = RandomForestClassifier(n_estimators=10, min_samples_split=2)

rf_result=rf.fit(X_train,y_train)

rf_pred = rf_result.predict(X_test)


# K-nearest neighbor: let us try a range of k to see what might be the best k
##k_range=range(1,30)
##
##scores=[]
##
##for k in k_range:
##
##    knn = KNeighborsClassifier(n_neighbors=k)
##
##    knn_result=knn.fit(X_train,y_train)
##
##    knn_pred = knn.predict(X_test)
##
##    scores.append(metrics.accuracy_score(y_test, knn_pred))

##plt.plot(k_range, scores)
##plt.xlabel('Value of k for KNN')
##plt.ylabel('Performance')
##plt.title('The effect of "k" in k-nearest neighbor')

knn = KNeighborsClassifier(n_neighbors=13)

knn_result=knn.fit(X_train,y_train)

knn_pred = knn.predict(X_test)

### Linear SVM:

clf = svm.LinearSVC(dual=False, C=100) #C=0.01,1,10,100 AUC=0.5

clf_result=clf.fit(X_train,y_train)

clf_pred=clf.predict(X_test)

### ROC plot preparation
fpr, tpr, thresholds =roc_curve(y_test, lr_pred) #roc_curve(true level,predicted outcome)
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

fpr2, tpr2, thresholds2 =roc_curve(y_test, rf_pred)
roc_auc2 = auc(fpr2,tpr2)
print("Area under the ROC curve : %f" % roc_auc2)

fpr3, tpr3, thresholds3 =roc_curve(y_test, knn_pred)
roc_auc3 = auc(fpr3, tpr3)
print("Area under the ROC curve : %f" % roc_auc3)

fpr4, tpr4, thresholds4 =roc_curve(y_test, clf_pred)
roc_auc4 = auc(fpr4, tpr4)
print("Area under the ROC curve : %f" % roc_auc4)

### Plot ROC plots
plt.figure()
plt.plot(fpr,tpr,label='Logistic Regression(AUC = %0.2f)' % roc_auc)
plt.plot(fpr2,tpr2,label='Random Forest(AUC = %0.2f)' % roc_auc2)
plt.plot(fpr3,tpr3,label='kNN, n=13(AUC = %0.2f)' % roc_auc3)
plt.plot(fpr4,tpr4,label='LinearSVM(AUC = %0.2f)' % roc_auc4)
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.plot([0, 1], [0, 1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


