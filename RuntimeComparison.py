import pandas
from sklearn import linear_model     
from sklearn.preprocessing import PolynomialFeatures     
from sklearn.svm import SVR   
from sklearn.cross_decomposition import PLSRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error
import random  
import math
import timeit
      
bioData = pandas.read_csv("data.csv")  
X = bioData[['Amylose', 'Glycerol']]     
y = bioData['TensileStrength']  
   
numOfIterations = 250  
   
rsquaredLinear = 0  
rmseLinear = 0  
rsquaredPolynomial = 0  
rmsePolynomial = 0  
rsquaredSVR = 0  
rmseSVR = 0  
rsquaredPLS = 0  
rmsePLS = 0  

linearListRSquared = []
polynomialListRSquared = []
svrListRSquared = []
plsListRSquared = []

linearListRMSE = []
polynomialListRMSE = []
svrListRMSE = []
plsListRMSE = []

startingTimeLin = timeit.default_timer()

for randomStateValue in range (numOfIterations):  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random.randrange(0, numOfIterations, 1))  
   
  linearReg = linear_model.LinearRegression()     
  linearReg.fit(X_train, y_train)  
  y_pred_Linear = linearReg.predict(X_test)  
  rsquaredLinear += linearReg.score(X_test, y_test) / numOfIterations
  rmseLinear += math.sqrt(mean_squared_error(y_test, y_pred_Linear)) / numOfIterations

print("Linear")  
print(rsquaredLinear)  
print(rmseLinear)  
print("") 

endingTimeLin = timeit.default_timer()
print("Linear Runtime:", endingTimeLin - startingTimeLin)

startingTimePoly = timeit.default_timer()

for randomStateValue in range (numOfIterations):  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random.randrange(0, numOfIterations, 1))  

  polynomialReg = PolynomialFeatures(degree=2)     
  X_polynomial = polynomialReg.fit_transform(X_train)     
  polynomialReg.fit(X_polynomial, y_train)     
  otherLinearReg = linear_model.LinearRegression()     
  otherLinearReg.fit(X_polynomial, y_train)   
  y_pred_Polynomial = otherLinearReg.predict(polynomialReg.fit_transform(X_test))  
  rsquaredPolynomial += otherLinearReg.score(polynomialReg.fit_transform(X_test), y_test) / numOfIterations
  rmsePolynomial += math.sqrt(mean_squared_error(y_test, y_pred_Polynomial)) / numOfIterations

print("Polynomial")  
print(rsquaredPolynomial)  
print(rmsePolynomial)  
print("")  

endingTimePoly = timeit.default_timer()
print("Polynomial Runtime:", endingTimePoly - startingTimePoly)

startingTimeSVR = timeit.default_timer()

for randomStateValue in range (numOfIterations):  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random.randrange(0, numOfIterations, 1))  
   
  supportVectorReg = SVR(kernel = 'rbf')   
  supportVectorReg.fit(X_train, y_train)  
  y_pred_SVR = supportVectorReg.predict(X_test)  
  rsquaredSVR += supportVectorReg.score(X_test, y_test) / numOfIterations
  rmseSVR += math.sqrt(mean_squared_error(y_test, y_pred_SVR)) / numOfIterations

print("SVR")  
print(rsquaredSVR)  
print(rmseSVR)  
print("")  

endingTimeSVR = timeit.default_timer()
print("SVR Runtime:", endingTimeSVR - startingTimeSVR)

startingTimePLS = timeit.default_timer()

for randomStateValue in range (numOfIterations):  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random.randrange(0, numOfIterations, 1))  

  n_components = 2  
  pls_model = PLSRegression(n_components = n_components)  
  pls_model.fit(X_train, y_train)  
   
  y_pred = pls_model.predict(X_test)  
  rsquaredPLS += pls_model.score(X_test, y_test) / numOfIterations
  rmsePLS += math.sqrt(mean_squared_error(y_test, y_pred)) / numOfIterations 
   
print("PLS")  
print(rsquaredPLS)  
print(rmsePLS)  
print("")

endingTimePLS = timeit.default_timer()
print("PLS Runtime:", endingTimePLS - startingTimePLS)