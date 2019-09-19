import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df=pd.read_csv('Ecommerce Customers')

# checking df
# print(df.head(),df.describe(),df.info)

# checking and analyzing relationships between different data
print(df.corr())
sns.set_style('whitegrid')
'''sns.pairplot(df)
# sns.jointplot(df['Time on Website'],df['Yearly Amount Spent'],s=15)
sns.jointplot(df['Time on App'],df['Length of Membership'],kind='hex')
sns.regplot(df['Length of Membership'],df['Yearly Amount Spent'], scatter_kws={"s": 10})
'''

# preparing data for testing

X=df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=42)

lm=LinearRegression()

#testing
lm.fit(X_train,y_train)
coefficients=pd.DataFrame(lm.coef_,index=['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership'],columns={'coef'})
# print(coefficients)

# checking how good is the model
prediction=lm.predict(X_test)
sns.scatterplot(y_test,prediction)

# calculating mean errors

mse = metrics.mean_squared_error(y_test,prediction)
mae = metrics.mean_absolute_error(y_test,prediction)
rmse = np.sqrt(mse)
error_df=pd.DataFrame.from_dict({'MSE':[mse],'MAE':[mae],'RMSE':[rmse]},orient='index',columns=['values'],)
print(error_df)

# checking residuals - distances betweent true and tested data
plt.figure()
plt.title('residual histogram')
residuals=sns.distplot(y_test-prediction)

plt.show()