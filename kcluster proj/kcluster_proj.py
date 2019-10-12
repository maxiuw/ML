import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs  # making artificial data
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# this is unsupervised learning algorithm created in order to assign students to
# proper colleges
# the main task of it is to recognize a pattern in the data and segregate it

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

df = pd.read_csv('College_Data')
print(df.head())
sns.set_style('whitegrid')
# sns.pairplot(df,hue='Private')
# dependencies
sns.lmplot('Room.Board', 'Grad.Rate', data=df, hue='Private', fit_reg=False, scatter_kws={"s": 5})
sns.lmplot('Outstate', 'F.Undergrad', data=df, hue='Private', fit_reg=False, scatter_kws={"s": 5})
plt.figure()

# tuition for outstate
df[df['Private'] == 'Yes']['Outstate'].plot(kind='hist', bins=30, color='red')
df[df['Private'] == 'No']['Outstate'].plot(kind='hist', bins=30, color='blue')

plt.figure()
# gradutation rate comparison


# checking uni name with max grad.rate
df[df['Private'] == 'Yes']['Grad.Rate'].plot(kind='hist', bins=30, color='red')
df[df['Private'] == 'No']['Grad.Rate'].plot(kind='hist', bins=30, color='blue')

pivot = pd.pivot_table(df, values='Grad.Rate', index='Unnamed: 0')
print(pivot.sort_values(by='Grad.Rate')[-1:])
df['Grad.Rate']['Cezenovia College'] = 100


# fitting the model
km = KMeans(n_clusters=2)
km.fit(df.drop(['Private', 'Unnamed: 0'], axis=1))
print(km.cluster_centers_)


def cluster(x):
    if x == 'Yes':
        return 1
    else:
        return 0


df['Cluster'] = df['Private'].apply(cluster)

print(df.head())

print(confusion_matrix(df['Cluster'],km.labels_))
print(classification_report(df['Cluster'],km.labels_))

plt.show()
