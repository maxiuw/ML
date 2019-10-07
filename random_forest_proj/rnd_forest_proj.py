# checking lendingclub.com data in order to decide if the person should get a loan
# credit.plicy 1 if customer meets credit underwriting 0 otherwise
# purpose the purpose of the loan
# int.rate - interest rate
# installment - monthly payment
# log.annual.inc ln of annual income of borrower
# dit amount of debt /income
# fico FICO credit score
# revol.bal - revolving balance of borrower
# inq.last6mnth no. of inquiries by borrower in last 6months
# deling 2year - how many times borrower was more that 30 ddays late in payment in last 2y
# pub.rec public rec
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

# data

df = pd.read_csv('loan_data.csv')
# print(df.head(),'\n',df.info(),'\n',df.describe())
sns.set_style("whitegrid")

'''plt.figure(figsize=(10,6))
df[df['credit.policy']==1]['fico'].hist(alpha=0.5,color='b',label='cred policy=1',bins=30)
df[df['credit.policy']==0]['fico'].hist(alpha=0.5,color='r',label='cred policy=0',bins=30)
plt.legend()
plt.xlabel('FICO')

plt.figure(figsize=(10,6))
df[df['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='b',label='cred policy=1',bins=30)
df[df['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='r',label='cred policy=0',bins=30)
plt.legend()
plt.xlabel('FICO')

plt.figure(figsize=(10,6))
sns.countplot(x=df['purpose'],hue=df['not.fully.paid'])
plt.figure(figsize=(10,6))
sns.jointplot(x='fico',y='int.rate',data=df)
'''
# making dummy var for purpose column
cat_feats = ['purpose']
final_df = pd.get_dummies(df,columns=cat_feats,drop_first=True)
# print(final_df.head())

# training data
X = final_df.drop('credit.policy',axis=1)
y=df['credit.policy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)

forest = RandomForestClassifier(n_estimators=300)
forest.fit(X_train,y_train)
prediction = forest.predict(X_test)

print(classification_report(prediction,y_test),'\n',
      confusion_matrix(prediction,y_test))




plt.show()