import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# defining if the client click or not on the advertisement

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

# data
df=pd.read_csv('advertising.csv')
print(df.head(5))
# print(df.describe())

# checking data
# sns.set_style('whitegrid')
# sns.distplot(df['Age'],kde=False,bins=30)
# sns.jointplot('Age','Area Income',data=df,s=10)
# sns.jointplot('Age','Daily Time Spent on Site',data=df,kind='kde',color='r')
# sns.jointplot('Daily Internet Usage','Daily Time Spent on Site',data=df,s=10,color='g')
# sns.pairplot(df,hue='Clicked on Ad',palette='bwr')

df=df.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1)

# training model

X=df.drop(['Clicked on Ad'],axis=1)
y=df['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)

# evaluating our model

print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


# roc curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()




# print(df.head())



plt.show()