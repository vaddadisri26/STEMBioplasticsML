import pandas
from sklearn import linear_model     
from sklearn.preprocessing import PolynomialFeatures     
from sklearn.svm import SVR   
from sklearn.cross_decomposition import PLSRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error
import random  
import math  
      
bioData = pandas.read_csv("data.csv")  
X = bioData[['Amylose', 'Glycerol']]     
y = bioData['TensileStrength']  

polynomialReg = PolynomialFeatures(degree=2)     
X_polynomial = polynomialReg.fit_transform(X)     
polynomialReg.fit(X_polynomial, y)     
otherLinearReg = linear_model.LinearRegression()     
otherLinearReg.fit(X_polynomial, y) 

temp = 0
max = 100
normalizedMaxAmylose = 0
normalizedMaxGlycerol = 0
optimalTensileStrength = 10

for i in range(21):  
    for j in range(21):
        temp = abs(optimalTensileStrength - (otherLinearReg.predict(polynomialReg.fit_transform([[j/20, i/20]])))[0])
        if (temp < max):
            max = temp
            maxAmylose = j/20
            maxGlycerol = i/20

print(maxAmylose * 11.16 + 0.56, ", ", maxGlycerol * 60, ", ", optimalTensileStrength - max)
''' 
11.16 <- the Max Amylose in Data - Min Amylose in Data.
0.56 <- the Min Amylose in Data.
60 <- the Max Amylose in Data - Min Amylose in Data.
These numbers are used to convert the normalized data back into genuine concentrations. 
Note that this does not account for when new data is added to the dataset.
A future step will be to account for this and make sure that the machine learning algorithm can run on its own given user-inputted data.
'''
