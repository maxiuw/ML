import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)


# data
df = pd.read_csv('KNN_Project_Data')
print(df.head())

# sns.pairplot(df,hue='TARGET CLASS')
# plt.show()
print(sns.__version__)

# standarizing variables to achieve similar range of values

standar = StandardScaler()

standar.fit(df.drop('TARGET CLASS',axis=1))


standard_df =standar.transform(df.drop('TARGET CLASS',axis=1))
stnd_df = pd.DataFrame(standard_df,columns=df.columns[:-1])
print(stnd_df.head())

# creating data for the model

X = stnd_df
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# choosing knn method and checking it out

knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))

# verifying the best neighbour size
# if mean pred error is different
# 7 seems like the best solution

error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    prediction1 = knn.predict(X_test)
    error_rate.append(np.mean(prediction1 != y_test))
print(error_rate)

plt.figure()
plt.plot(range(1,40),error_rate,color='r',linestyle='dashed',marker='o')
plt.show()