from IPython.display import Image
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

# program recognizing different spieces of iris


# checking images of different flowers
# The Iris Setosa
'''url1 = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
iris_s = Image(url1,width=300, height=300)

# The Iris Versicolor
from IPython.display import Image
url2 = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
iris_V = Image(url2,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url3 = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
iris_Vi = Image(url3,width=300, height=300)'''

# import some data
iris = sns.load_dataset('iris')
# creating df from the iris data
print(iris.keys())
df = pd.DataFrame(iris)
print(df.head())

# checking data dependency

# sns.pairplot(df,hue='species')
plt.figure()
plt.subplot(1, 3, 1)
sns.kdeplot(df[df['species'] == 'setosa']['sepal_length'], df[df['species'] == 'setosa']['sepal_width'],
            cmap='gist_ncar', shade=True, shade_lowest=False)
plt.subplot(1, 3, 2)
sns.kdeplot(df[df['species'] == 'virginica']['sepal_length'], df[df['species'] == 'virginica']['sepal_width'],
            cmap='rainbow', shade=True, shade_lowest=False)
plt.subplot(1, 3, 3)
sns.kdeplot(df[df['species'] == 'versicolor']['sepal_length'], df[df['species'] == 'versicolor']['sepal_width'],
            cmap='gnuplot', shade=True, shade_lowest=False)



# spliting data for traininig
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# creating model and training
parameters = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
svc = SVC()
svc.fit(X_train, y_train)
prediction = svc.predict(X_test)

# checking different model parameter
grid = GridSearchCV(SVC(),parameters,verbose=3)
grid.fit(X_train,y_train)

print(grid.best_estimator_,'\n',grid.best_params_)

prediction_tuned = grid.predict(X_test)

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
print('\n after tuning \n')
print(confusion_matrix(y_test,prediction_tuned))
print(classification_report(y_test, prediction_tuned))
plt.show()
