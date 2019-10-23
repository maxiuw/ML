import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.pipeline import Pipeline

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

# data
df = pd.read_csv('yelp.csv')
# print(df.head(5),df.info(),'\n',df.describe())

# checking different features of data and
# text length dependency on the stars review

df['text length']=df['text'].apply(len)
# print(df['text length'])

'''g = sns.FacetGrid(df,col='stars')
g.map(plt.hist,'text length')

plt.figure()
sns.boxplot(x='stars',y='text length',data=df)

plt.figure()
sns.countplot(x='stars',data=df)'''

stars_mean = df.groupby('stars')
# print(stars_mean.mean(),'\n')
# print(stars_mean.mean().corr())

# sns.heatmap(data=stars_mean.mean().corr(),cmap='cool',annot=True)

yelp_class= df[(df['stars']==1)|(df['stars']==5)]
# print(yelp_class[yelp_class['stars']==1])

X = yelp_class['text']
y = yelp_class['stars']

# tarnsforming text into a data
X = CountVectorizer().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
nb = MultinomialNB()
nb.fit(X_train,y_train)
prediction = nb.predict(X_test)

# analysing the results
print(confusion_matrix(y_test,prediction),'\n',classification_report(y_test,prediction))

# improving results using tfidf and pipeline

pipe = Pipeline(steps=[('cout vector',CountVectorizer()),('tfidf',TfidfTransformer()),('fiting',MultinomialNB())])
X2 = yelp_class['text']
y2 = yelp_class['stars']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=101)
pipe.fit(X_train2,y_train2)
prediction2 = pipe.predict(X_test2)

# unfortunatelly tfidf made the resutls worse
print(confusion_matrix(y_test2,prediction2),'\n',classification_report(y_test2,prediction2))


plt.show()