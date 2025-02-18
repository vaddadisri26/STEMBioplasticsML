import pandas
from sklearn import linear_model     
from sklearn.preprocessing import PolynomialFeatures     
from sklearn.svm import SVR   
from sklearn.cross_decomposition import PLSRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error
import random  
import math
import scipy.stats 
import numpy as np
      
bioData = pandas.read_csv("data.csv")  
X = bioData[['Amylose', 'Glycerol']]     
y = bioData['TensileStrength']  
   
numOfIterations = 250  

linearListRSquared = []
polynomialListRSquared = []
svrListRSquared = []
plsListRSquared = []

linearListRMSE = []
polynomialListRMSE = []
svrListRMSE = []
plsListRMSE = []
   
for randomStateValue in range (numOfIterations):  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random.randrange(0, numOfIterations, 1))  
   
  linearReg = linear_model.LinearRegression()     
  linearReg.fit(X_train, y_train)  
  y_pred_Linear = linearReg.predict(X_test)  
  linearListRSquared.append(linearReg.score(X_test, y_test))
  linearListRMSE.append(math.sqrt(mean_squared_error(y_test, y_pred_Linear)))
   
  polynomialReg = PolynomialFeatures(degree=2)     
  X_polynomial = polynomialReg.fit_transform(X_train)     
  polynomialReg.fit(X_polynomial, y_train)     
  otherLinearReg = linear_model.LinearRegression()     
  otherLinearReg.fit(X_polynomial, y_train)   
  y_pred_Polynomial = otherLinearReg.predict(polynomialReg.fit_transform(X_test))  
  polynomialListRSquared.append(otherLinearReg.score(polynomialReg.fit_transform(X_test), y_test))
  polynomialListRMSE.append(math.sqrt(mean_squared_error(y_test, y_pred_Polynomial)))
   
  supportVectorReg = SVR(kernel = 'rbf')   
  supportVectorReg.fit(X_train, y_train)  
  y_pred_SVR = supportVectorReg.predict(X_test)  
  svrListRSquared.append(supportVectorReg.score(X_test, y_test))
  svrListRMSE.append(math.sqrt(mean_squared_error(y_test, y_pred_SVR)))
   
  n_components = 2  
  pls_model = PLSRegression(n_components = n_components)  
  pls_model.fit(X_train, y_train)  
   
  y_pred = pls_model.predict(X_test)  
  plsListRSquared.append(pls_model.score(X_test, y_test))
  plsListRMSE.append(math.sqrt(mean_squared_error(y_test, y_pred)))

tPolynomialLinearRsquared, pPolynomialLinearRsquared = scipy.stats.ttest_ind(a = polynomialListRSquared, b = linearListRSquared, equal_var = False)
print("PolynomialLinearRSquared T-stat:", tPolynomialLinearRsquared)
print("PolynomialLinearRSquared P-value:", pPolynomialLinearRsquared)
print("")

tPolynomialSVRRsquared, pPolynomialSVRRsquared = scipy.stats.ttest_ind(a = polynomialListRSquared, b = svrListRSquared, equal_var = False)
print("PolynomialSVRRSquared T-stat:", tPolynomialSVRRsquared)
print("PolynomialSVRRSquared P-value:", pPolynomialSVRRsquared)
print("")

tPolynomialPLSRsquared, pPolynomialPLSRsquared = scipy.stats.ttest_ind(a = polynomialListRSquared, b = plsListRSquared, equal_var = False)
print("PolynomialPLSRsquared T-stat:", tPolynomialPLSRsquared)
print("PolynomialPLSRsquared P-value:", pPolynomialPLSRsquared)
print("")

tPolynomialLinearRMSE, pPolynomialLinearRMSE = scipy.stats.ttest_ind(a = polynomialListRMSE, b = linearListRMSE, equal_var = False)
print("PolynomialLinearRMSE T-stat:", tPolynomialLinearRMSE)
print("PolynomialLinearRMSE P-value:", pPolynomialLinearRMSE)
print("")

tPolynomialSVRRMSE, pPolynomialSVRRMSE = scipy.stats.ttest_ind(a = polynomialListRMSE, b = svrListRMSE, equal_var = False)
print("PolynomialSVRRMSE T-stat:", tPolynomialSVRRMSE)
print("PolynomialSVRRMSE P-value:", pPolynomialSVRRMSE)
print("")

tPolynomialPLSRMSE, pPolynomialPLSRMSE = scipy.stats.ttest_ind(a = polynomialListRMSE, b = plsListRMSE, equal_var = False)
print("PolynomialPLSRMSE T-stat:", tPolynomialPLSRMSE)
print("PolynomialPLSRMSE P-value:", pPolynomialPLSRMSE)
print("")