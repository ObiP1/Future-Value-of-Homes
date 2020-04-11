import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from matplotlib import ticker
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
%matplotlib inline   
import pandas as pd 

#Scaling overall size
rcParams['figure.figsize'] = 20,10
#Grabbing our dataset
df = pd.read_csv('https://raw.githubusercontent.com/ObiP1/The-Future-Value-of-Homes/master/AverageHomeValues.csv')
df[df.columns[1:]] = df[df.columns[1:]].replace('[\$,]', '', regex=True).astype(float)

#Titles 
plt.title('Median Cost Of Maryland Homes', fontsize=30)
plt.ylabel('Median Price Of Home',fontsize=25)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
plt.xlabel('Year', fontsize=25)

#implements the use of our data to forecast future prices
data_train = df[df.YEAR < 2020]
data_test = df[df.MED_COST >= 2020]

X_train,X_test,y_train,y_test=train_test_split(df.YEAR,df.MED_COST)
X_All = df.YEAR[:, np.newaxis]

X1=X_train
X2=X_test

#Mandatory Reshaping
X_train = X1.values.reshape(-1, 1)
X_test = X2.values.reshape(-1, 1)

#Fitting and Predicting
bayesian_reg = BayesianRidge().fit(X_train,y_train)

# Prediction of future values
X_predict = df['YEAR'].append(pd.Series(range(2030, 2050, 10)))[:, np.newaxis]
pred_lr = bayesian_reg.predict(X_predict)  

# Display of Accuraccy 
print("Accuracy on training set: {:.3f}".format(bayesian_reg.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(bayesian_reg.score(X_test,y_test)))

#Plotting our Data to plot
plt.plot(data_train.YEAR, data_train.MED_COST, label="Training Data")
plt.plot(data_test.YEAR, data_test.MED_COST, label="Test Data", linestyle=':')
plt.plot(X_predict, pred_lr, label="Bayesian Prediction", linestyle='--')
plt.legend()

# Axis increments and implementation of grid
plt.grid(True)
plt.yticks([0,50000,100000,150000,200000,250000,300000,350000,400000,450000])
plt.xticks([1940,1950,1960,1970,1980,1990,2000,2010,2020,2030,2040,2050])
plt.show()
