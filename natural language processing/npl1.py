import nltk
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix


# uci data sets
# sms spam collection data set
# spam detection filter
# nltk.download_shell() # downloading sms data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

# checking the content
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection',encoding='utf8')]
# print(len(messages))
# print(messages[50])
# for mess_no,msg in enumerate(messages[:10]):
#     print(mess_no,msg)

# transforming messages into DF
messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])
# print(mesg.describe())

# print(messages.groupby('label').describe()) # grouping by the label

# feature engineering
# checking which properties can help us to divide the ham vs. spam
# length seems like a good one

messages['length'] = messages['message'].apply(len)
# print(messages)

# messages['length'].plot.hist(bins=150)
# print(messages)
# print(messages[messages['length']==910]['message'].iloc[0])
'''plt.figure()
messages.hist(column='length',by='label',bins=60,figsize=(12,4))
plt.figure()
messages[messages['label']=='ham']['length'].hist(bins=150,color='red')
messages[messages['label']=='spam']['length'].hist(bins=150,color='blue')'''

'''# cleaning stop words, making the msg easier  to read
mess = 'asms! notice: punt...'
nonpunc = [c for c in mess if c not in string.punctuation] # nonpunc = [c for c in mess if c!=' ']
nonpunc = ''.join(nonpunc) # joining back elements in the list back together
print(nonpunc.split())
clean = [word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]
print(clean)'''

# f to clean the punct and stop words
def cleaning(message):
    message_nopunc = [c for c in message if c not in string.punctuation]
    message_nostop = ''.join(message_nopunc)
    message_final = [word for word in message_nostop.split() if word not in stopwords.words('english')]

    return message_final

# tokenizing the dta, converting normal txt strings into a list of tokens

# print(messages['message'].head(5).apply(cleaning))

# vectorization
# converting the message into a vector which scilearn algorithm can work with
# 'bag of words' model'
# 1 count how any times does a word occurs in each msg
# 2 weigh the counts, freq tokens (words) get lower weight
# 3 normalize the vectors to the unit length, to abstract from the original txt length

# 1
# creating a big matrix of messages x words to count how many times each word was used
bag_ow_transformer = CountVectorizer(analyzer=cleaning).fit(messages['message'])
#print(bag_ow_transformer.vocabulary_)

mess4 = messages['message'][3]
bow4 = bag_ow_transformer.transform([mess4])
print(bow4,'\n',bow4.shape) # 7 unique words in this sentence and ho many times do they appear
# to see what is the world we have to grab the word index and use it here
print(bag_ow_transformer.get_feature_names()[9554]) # this word is 'reschedule'

messages_bow = bag_ow_transformer.transform(messages['message']) # matrix 5572 x 11425
print((messages_bow.nnz)) # no zeros information

# comapring the ratio of no zero messages to zero messages
sparsity = (100.0*messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))
print(sparsity) # around 0.079 nz message/total number of messages

# scaling down the impact tocken that occur very frequently in a given corpuse
# term frequency instead of document frequency
# tf-idf => unique word count *log(total messages/number of messages where the word appears)
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

# e.g. checking the frq of a random word university
print(tfidf_transformer.idf_[bag_ow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)

# naive based classifier to classify the messages
# based on tfidf model
# this model however is not so accurate for this set of data cause we used all data
# which we had, for training
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])
print(spam_detect_model.predict(tfidf4)[0]) # predicting if the following msg is ham or spam

msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)

# summarizing all the steps into a pipeline so there is no need of repeating the same process all the time
# for different sets of data
# pipeline takes 'steps' arg
# 1st tuple - name of the step
pipeline = Pipeline([('bow',CountVectorizer(analyzer=cleaning)),('tfidf',TfidfTransformer()),('classifier',MultinomialNB())])
pipeline.fit(msg_train,label_train)
prediction = pipeline.predict(msg_test)

print(classification_report(label_test,prediction),confusion_matrix(label_test,prediction))

# classification report of the model on a true test data





plt.show()