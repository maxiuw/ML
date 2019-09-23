import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# setting bigger range of data displaying

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

# data
df_train=pd.read_csv('titanic_train.csv')
# print(df_train.head())
# print(df_train.isnull()) # True if data is unknown

# showing where data is missing
# cabing is missing too much data, age misses some data


# ploting the data
sns.set_style('whitegrid')
# sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.figure()

# sns.countplot(x='Survived',data=df_train,hue='Sex')
# it seems like female were more probable to survive

# sns.countplot(x='Survived',data=df_train,hue='Pclass')
# sns.distplot(df_train['Age'].dropna(),kde=False,bins=30)

# checking if they have kids (most didnt)
# sns.countplot(x='SibSp',data=df_train)

# filling in the missing data with average data (e.g. age with avg age)

# sns.boxplot(x='Pclass',y='Age',data=df_train)


# if the age is missing returning avg age of the class, otherwise returning age
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        elif Pclass==3:
            return 24
    else:
        return Age

# applying function to append isnull values
df_train['Age']=df_train[['Age','Pclass']].apply(impute_age,axis=1)



# dropping cabin col due to too big number of information missing

df_train.drop('Cabin',axis=1,inplace=True)

# dropping any missing information (one left)
df_train.dropna(inplace=True)


# creating dummy var - replacing strings (e.g. male/Female or name of the city)
# we have to drop one of the columns, cause otherwise we would have female and male
# one will be 0 and the other one would be always the opposite

sex_col = pd.get_dummies(df_train['Sex'],drop_first=True)
# print(pd.get_dummies(df_train['Sex']))

embark = pd.get_dummies(df_train['Embarked'],drop_first=True)

# adding the columns to our df

df_train=pd.concat([df_train,sex_col,embark],axis=1)
# print(df_train.head())

# dropping columns which we r not going to use
# cause they r hard to implement into ml algorith
df_train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

# passenger ID is just a number as index so we r dropping it as well
df_train.drop('PassengerId',axis=1,inplace=True)

print(df_train.tail())

# training

X=df_train.drop('Survived',axis=1)
y=df_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)


# rating our model
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))






#plt.figure()



# sns.heatmap(df_train.isnull(),yticklabels=False,xticklabels=False)

plt.show()