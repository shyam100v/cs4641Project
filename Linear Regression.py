#!/usr/bin/env python
# coding: utf-8

# In[2]:


# LINEAR REGRESSSION
#Video used: https://www.youtube.com/watch?v=KKmu960FS2Y&t=274s

import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
import csv
import seaborn as sns
import statsmodels.api as sm
import math

from scipy import stats
from scipy.stats import kurtosis, skew
get_ipython().run_line_magic('matplotlib', 'inline')

#filein ='oldAndNewOnlyEnglish_noDuplicatesPlus1.csv'

filein = 'oldAndNewOnlyEnglish_noDuplicatesWithLikesAndComments.csv'
df = pd.read_csv(filein)

#Drop any missing values
#display(df.isna().any())
#df = df.dropna()
#display(df.isna().any())


# In[3]:


views = df["videoViews"]
#likes = df["VideoLikesPlus1"]
likes = df["videoLikes"]
dislikes = df["videoDislikes"]
comments = df["videoCommentCount"]
categoryID = df["videoCategoryId"]
publishTime = df["publishedZTimeFloat"]
dayDifference = df["dayDifference"]
dayofWeek = df["publishedDayOfWeek"]
titles = df["videoTitle"]

#get different title analysis
titleLengths = titles.str.len()
titleLengths = titleLengths.rename('titleLength')
numCaps = titles.str.count(r'[A-Z]')
numCaps = numCaps.rename('numCaps')
numPuncs = titles.str.count('!') + titles.str.count('\?')
numPuncs = numPuncs.rename('numPuncs')

df = pd.concat([views, likes, dislikes,comments, categoryID, publishTime,dayDifference,dayofWeek,titles,titleLengths, numCaps, numPuncs],axis = 1)

X = likes + 1
print(X)
Y = views
plt.plot(X, Y, 'o', color = 'cadetblue', label = 'ViewsVersusLikes')
plt.title('Views versus likes')
plt.xlabel('Likes')
plt.ylabel('Views')
plt.legend()
plt.show()

df.corr()

#Check if there are values = 0
#display(likes.eq(0).any().any())
#display(dislikes.eq(0).any().any())


# In[4]:


#Statistical summary
#useful to see if there any outliers and see if the data falls within
# 3 standard deviations or not
df.describe()


# In[6]:


#REMOVING OUTLIERS OF LIKES, DISLIKES AND VIEWS
#Doing the histogram shows that there is many outliers on the data
views_array = views.to_numpy()
dislikes_array = dislikes.to_numpy()
likes_array = likes.to_numpy()
titleLengths_array = titleLengths.to_numpy()

# Remove outliers views
meanviews = np.mean(views_array)
standard_deviation = np.std(views_array)
distance_from_mean = abs(views_array - meanviews)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
views_array = views_array[not_outlier]

#Apply modification to the rest of variables
dislikes_array = dislikes_array[not_outlier]
likes_array = likes_array[not_outlier]
titleLengths_array = titleLengths_array[not_outlier]

#---------------------------------------------------------------------------
#Remove outliers dislikes
meandislikes = np.mean(dislikes_array)
standard_deviation = np.std(dislikes_array)
distance_from_mean = abs(dislikes_array - meandislikes)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
dislikes_array = dislikes_array[not_outlier]


#Apply modification to the rest of variables
likes_array = likes_array[not_outlier]
views_array = views_array[not_outlier]
titleLengths_array = titleLengths_array[not_outlier]


#-----------------------------------------------------------------------
#Remove outliers likes
meanlikes = np.mean(likes_array)
standard_deviation = np.std(likes_array)
distance_from_mean = abs(likes_array - meanlikes)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
likes_array = likes_array[not_outlier]


#Apply modification to the rest of variables
dislikes_array = dislikes_array[not_outlier]
views_array = views_array[not_outlier]
titleLengths_array = titleLengths_array[not_outlier]


# In[149]:


#HISTOGRAM OF VIEWS AFTER REMOVING OUTLIERS
views_series_no_outliers = pd.Series(views_array)
#views.hist(grid=True, color = 'cadetblue') #Old data histogram
views_series_no_outliers.hist() #After removing outliers histogram


# In[109]:


#HISTOGRAM OF DISLIKES AFTER REMOVING OUTLIERS
dislikes_series_no_outliers = pd.Series(dislikes_array)
#dislikes.hist(grid=True, color = 'cadetblue') #Old data histogram
dislikes_series_no_outliers.hist() #After removing outliers histogram


# In[110]:


#HISTOGRAM OF LIKES AFTER REMOVING OUTLIERS
likes_series_no_outliers = pd.Series(likes_array)
#likes.hist(grid=True, color = 'cadetblue') #Old data histogram
likes_series_no_outliers.hist()


# In[111]:


#LOG OF VIEWS
log_views_array = np.log(views_array)
views_series_no_outliers_log = pd.Series(log_views_array)
views_series_no_outliers_log.hist()


# In[113]:


log_dislikes_array = np.log(dislikes_array)
dislikes_series_no_outliers_log = pd.Series(log_dislikes_array)
dislikes_series_no_outliers_log.hist()


# In[114]:


#LOG OF LIKES
log_likes_array = np.log(likes_array)
likes_series_no_outliers_log = pd.Series(log_likes_array)
likes_series_no_outliers_log.hist()


# In[120]:


#Scatter plot of the log of views versus the log of likes
X = likes_series_no_outliers_log
Y = views_series_no_outliers_log
plt.plot(X, Y, 'o', color = 'cadetblue', label = 'ViewsVersusLikes')
plt.title('Log Views versus log likes')
plt.xlabel('Log Likes')
plt.ylabel('Log Views')
plt.legend()
plt.show()


# In[158]:


#Scatter plot of the log of views versus the log of dislikes
X = dislikes_series_no_outliers_log
Y = views_series_no_outliers_log
plt.plot(X, Y, 'o', color = 'cadetblue', label = 'ViewsVersusDisLikes')
plt.title('Log Views versus log dislikes')
plt.xlabel('Log Disikes')
plt.ylabel('Log Views')
plt.legend()
plt.show()


# In[159]:


#Check if data is skewed BEFORE LOG
likes_kurtosis = kurtosis(likes_series_no_outliers, fisher = True)
dislikes_kurtosis = kurtosis(dislikes_series_no_outliers, fisher = True)
views_kurtosis = kurtosis(views_series_no_outliers, fisher = True)

# calculate the skewness
likes_skew = skew(likes_series_no_outliers)
dislikes_skew = skew(dislikes_series_no_outliers)
views_skew = skew(views_series_no_outliers)

display ("BEFORE APPLYING THE LOG")
display("Likes Excess Kurtosis: {:.2}".format(likes_kurtosis))  # this looks fine
display("Dislikes Excess Kurtosis: {:.2}".format(dislikes_kurtosis))  # this looks fine
display("Views Excess Kurtosis: {:.2}".format(views_kurtosis))  # this looks fine

display("Likes Skew: {:.2}".format(likes_skew)) # moderately skewed
display("Dislikes Skew: {:.2}".format(dislikes_skew)) # moderately skewed
display("Views Skew: {:.2}".format(views_skew)) # moderately skewed, it's a little high but we will accept it.


display("All 3 Kurtisus are leptokurtic - Tails are longer and fatter, and the central peak is higher and shaper")
display("All 3 distributions are highly skewed")


# In[160]:


#Check if data is skewed AFTER LOG
likes_kurtosis = kurtosis(likes_series_no_outliers_log, fisher = True)
dislikes_kurtosis = kurtosis(dislikes_series_no_outliers_log, fisher = True)
views_kurtosis = kurtosis(views_series_no_outliers_log, fisher = True)

# calculate the skewness
likes_skew = skew(likes_series_no_outliers_log)
dislikes_skew = skew(dislikes_series_no_outliers_log)
views_skew = skew(views_series_no_outliers_log)

display("AFTER APPLYING THE LOG")
display("Likes Excess Kurtosis: {:.2}".format(likes_kurtosis))  # this looks fine
display("Dislikes Excess Kurtosis: {:.2}".format(dislikes_kurtosis))  # this looks fine
display("Views Excess Kurtosis: {:.2}".format(views_kurtosis))  # this looks fine

display("Likes Skew: {:.2}".format(likes_skew)) # moderately skewed
display("Dislikes Skew: {:.2}".format(dislikes_skew)) # moderately skewed
display("Views Skew: {:.2}".format(views_skew)) # moderately skewed, it's a little high but we will accept it.


display("All 3 Kurtisus are platykurtic- Tails are shorter and thinner, and often its central peak is lower and broader.")
display("For dislikes and views, the distribution is approx symmetric. For likes, it is moderatelyt skewed")


# In[122]:


#Perform a kurtosis test before applyting the log
display("BEFORE THE LOG")
display('Likes')
display(stats.kurtosistest(likes_series_no_outliers))
display('Disikes')
display(stats.kurtosistest(dislikes_series_no_outliers))
display('Views')
display(stats.kurtosistest(views_series_no_outliers))

# perform a skew test
display('Likes')
display(stats.skewtest(likes_series_no_outliers))
display('Dislikes')
display(stats.skewtest(dislikes_series_no_outliers))
display('Views')
display(stats.skewtest(views_series_no_outliers))


# In[ ]:


#Perform a kurtosis test after applyting the log
display("AFTER THE LOG")
display('Likes')
display(stats.kurtosistest(likes_series_no_outliers_log))
display('Disikes')
display(stats.kurtosistest(dislikes_series_no_outliers_log))
display('Views')
display(stats.kurtosistest(views_series_no_outliers_log))

# perform a skew test
display('Likes')
display(stats.skewtest(likes_series_no_outliers_log))
display('Dislikes')
display(stats.skewtest(dislikes_series_no_outliers_log))
display('Views')
display(stats.skewtest(views_series_no_outliers_log))


# In[124]:


# BEFORE APPLYING THE LOG
#Split data in 80/20
X = likes_series_no_outliers
Y = views_series_no_outliers
x = X.to_numpy()
y = Y.to_numpy()
x_fixed = np.trunc(x.reshape(-1, 1))
y_fixed = np.trunc(y)
x_fixed2 = np.nan_to_num(x_fixed)
y_fixed2 = np.nan_to_num(y_fixed)
print(x_fixed2)
print(y_fixed2)
X_train, X_test, Y_train, Y_test = train_test_split(x_fixed, y_fixed, test_size=0.2, random_state = 1)
regression_model = LinearRegression()
#Build training model
regression_model.fit(X_train, Y_train)

#Explore the output
intercept = regression_model.intercept_
coefficient = regression_model.coef_[0]
print("BEFORE APPLYING THE LOG")
print("The coefficient for our model is {:.2}".format(coefficient))
print("The intercept for our model is {:.4}".format(intercept))
#prediction = regression_model.predict([[67.33]])
#predicted_vaue = prediction[0][0]


#Apply trained model to make prediction
#Y_pred = model.predict(X_test)

#Print model performance
#print('Coefficients:', model.coef_)
#print('Intercept:', model.intercept_)
#print('Mean squared error (MSE): %.2f'
      #% mean_squared_error(Y_test, Y_pred))
#print('Coefficient of determination (R^2): %.2f'
      #% r2_score(Y_test, Y_pred))

#Scatter plots
#sns.scatterplot(Y_test, Y_pred)


# In[134]:


# AFTER APPLYING THE LOG
#Split data in 80/20
X_log = likes_series_no_outliers_log
Y_log = views_series_no_outliers_log
x_log = X_log.to_numpy()
y_log = Y_log.to_numpy()
x_fixed_log = np.trunc(x_log.reshape(-1, 1))
y_fixed_log = np.trunc(y_log)
x_fixed2_log = np.nan_to_num(x_fixed_log)
y_fixed2_log = np.nan_to_num(y_fixed_log)
print(x_fixed2_log)
print(y_fixed2_log)
X_train_log, X_test_log, Y_train_log, Y_test_log = train_test_split(x_fixed_log, y_fixed_log, test_size=0.2, random_state = 1)
regression_model_log = LinearRegression()
#Build training model
regression_model_log.fit(X_train_log, Y_train_log)

#Explore the output
intercept_log = regression_model.intercept_
coefficient_log = regression_model.coef_[0]
print("AFTER APPLYING THE LOG")
print("The coefficient for our model is {:.2}".format(coefficient_log))
print("The intercept for our model is {:.4}".format(intercept_log))


# In[135]:


#Taking a single prediction BEFORE THE LOG
numberLikes = 10000
prediction = regression_model.predict([[numberLikes]])
predicted_value = prediction[0]
print("The predicted value is {:.4} number of views with {:} number of likes".format(predicted_value, numberLikes))


# In[136]:


#Making multiple predictions at once BEFORE THE LOG
y_predict = regression_model.predict(X_test)


# In[137]:


#Making multiple predictions at once AFTER THE LOG
y_predict_log = regression_model_log.predict(X_test_log)


# In[138]:


#Evaluate the model BEFORE THE LOG

#define our input
X = likes_series_no_outliers
Y = views_series_no_outliers
X2 = sm.add_constant(X)

#create a OLS model
model = sm.OLS(Y, X2)

# fit the data
est = model.fit()


# In[139]:


#Confidence Interval (95% CI)
est.conf_int()

#This demonstrates that the coefficient that determines the number of likes  exits between
#30.27 and 30.991


# In[150]:


#Evaluate the model AFTER THE LOG

#define our input
X_log = likes_series_no_outliers_log
Y_log = views_series_no_outliers_log
X2_log = sm.add_constant(X_log)

#create a OLS model
model_log = sm.OLS(Y_log, X2_log)

# fit the data
est_log = model_log.fit()


# In[151]:


#Confidence Interval (95% CI)
est_log.conf_int()


# In[26]:


#Calculate the mean squared error: punished larger errors
model_mse = mean_squared_error(Y_test, y_predict)

# Calculate the mean absolute error: gives an idea of magnitude byt no idea of direction
model_mae = mean_absolute_error(Y_test, y_predict)

# Calualte the root mean squared error
model_rmse = math.sqrt(model_mse)

#Display the output
print("MSE = {:.3}".format(model_mse))
print("MSE = {:.3}".format(model_mae))
print("RMSE = {:.3}".format(model_rmse))


# In[152]:


#AFTER APPLYING THE LOG
#Calculate the mean squared error: punished larger errors
model_mse_log = mean_squared_error(Y_test_log, y_predict_log)

# Calculate the mean absolute error: gives an idea of magnitude byt no idea of direction
model_mae_log = mean_absolute_error(Y_test_log, y_predict_log)

# Calualte the root mean squared error
model_rmse_log = math.sqrt(model_mse_log)

#Display the output
print("MSE = {:.3}".format(model_mse_log))
print("MSE = {:.3}".format(model_mae_log))
print("RMSE = {:.3}".format(model_rmse_log))


# In[153]:


#R-squared
model_r2 = r2_score(Y_test, y_predict)
print("R2: {:.2}".format(model_r2))


# In[155]:


#R-squared after log
model_r2_log = r2_score(Y_test_log, y_predict_log)
print("R2: {:.2}".format(model_r2_log))


# In[86]:


#Summary of the model
print(est.summary())


# In[156]:


#Plot the residuals and check that they are normally distributed
(Y_test - y_predict).hist(grid = False, color = 'royalblue')
plt.title("Model Residuals")
plot.show()


# In[90]:


#Plotting our line BEFORE THE LOG

# Plot outputs
plt.scatter(X_test, Y_test,  color='gainsboro', label = 'Data points')
plt.plot(X_test, y_predict, color='royalblue', linewidth = 3, linestyle= '-',label ='Regression Line')

plt.title("Linear Regression of Views versus Likes")
plt.xlabel("Likes")
plt.ylabel("Views")
plt.legend()
plt.show()

# The coefficients
print('Likes coefficient:' + '\033[1m' + '{:.2}''\033[0m'.format(regression_model.coef_[0]))

# The mean squared error
print('Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(model_mse))

# The mean squared error
print('Root Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(math.sqrt(model_mse)))

# Explained variance score: 1 is perfect prediction
print('R2 score: '+ '\033[1m' + '{:.2}''\033[0m'.format(r2_score(Y_test,y_predict)))


# In[157]:


#Plotting our line AFTER THE LOG

# Plot outputs
plt.scatter(X_test_log, Y_test_log,  color='gainsboro', label = 'Data points')
plt.plot(X_test_log, y_predict_log, color='royalblue', linewidth = 3, linestyle= '-',label ='Regression Line')

plt.title("Linear Regression of log Views versus log Likes")
plt.xlabel("Likes")
plt.ylabel("Views")
plt.legend()
plt.show()

# The coefficients
print('Likes coefficient:' + '\033[1m' + '{:.2}''\033[0m'.format(coefficient_log))

# The mean squared error
print('Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(model_mse_log))

# The mean squared error
print('Root Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(math.sqrt(model_mse_log)))

# Explained variance score: 1 is perfect prediction
print('R2 score: '+ '\033[1m' + '{:.2}''\033[0m'.format(r2_score(Y_test_log,y_predict_log)))


# NOW REPEAT PROCESS WITH DISLIKES
# 
# 
# 

# In[161]:


# AFTER APPLYING THE LOG
#Split data in 80/20
X_log_dis = dislikes_series_no_outliers_log
Y_log_dis = views_series_no_outliers_log
x_log_dis = X_log_dis.to_numpy()
y_log_dis = Y_log_dis.to_numpy()
x_fixed_log_dis = np.trunc(x_log_dis.reshape(-1, 1))
y_fixed_log_dis = np.trunc(y_log_dis)
x_fixed2_log_dis = np.nan_to_num(x_fixed_log_dis)
y_fixed2_log_dis = np.nan_to_num(y_fixed_log_dis)
print(x_fixed2_log_dis)
print(y_fixed2_log_dis)
X_train_log_dis, X_test_log_dis, Y_train_log_dis, Y_test_log_dis = train_test_split(x_fixed_log_dis, y_fixed_log_dis, test_size=0.2, random_state = 1)
regression_model_log_dis = LinearRegression()
#Build training model
regression_model_log_dis.fit(X_train_log_dis, Y_train_log_dis)

#Explore the output
intercept_log_dis = regression_model.intercept_
coefficient_log_dis = regression_model.coef_[0]
print("AFTER APPLYING THE LOG")
print("The coefficient for our model is {:.2}".format(coefficient_log_dis))
print("The intercept for our model is {:.4}".format(intercept_log_dis))


# In[162]:


#Making multiple predictions at once AFTER THE LOG - DISLIKES
y_predict_log_dis = regression_model_log_dis.predict(X_test_log_dis)


# In[163]:


#Evaluate the model AFTER THE LOG - DISLIKES

#define our input
X2_log_dis = sm.add_constant(X_log_dis)

#create a OLS model
model_log_dis = sm.OLS(Y_log_dis, X2_log_dis)

# fit the data
est_log_dis = model_log_dis.fit()


# In[166]:


#AFTER APPLYING THE LOG
#Calculate the mean squared error: punished larger errors
model_mse_log_dis = mean_squared_error(Y_test_log_dis, y_predict_log_dis)

# Calculate the mean absolute error: gives an idea of magnitude byt no idea of direction
model_mae_log_dis = mean_absolute_error(Y_test_log_dis, y_predict_log_dis)

# Calualte the root mean squared error
model_rmse_log_dis = math.sqrt(model_mse_log_dis)

#Display the output
print("MSE = {:.3}".format(model_mse_log_dis))
print("MSE = {:.3}".format(model_mae_log_dis))
print("RMSE = {:.3}".format(model_rmse_log_dis))


# In[167]:


#R-squared after log
model_r2_log_dis = r2_score(Y_test_log_dis, y_predict_log_dis)
print("R2: {:.2}".format(model_r2_log_dis))


# In[168]:


#Plotting our line AFTER THE LOG - DISLIKES

# Plot outputs
plt.scatter(X_test_log_dis, Y_test_log_dis,  color='gainsboro', label = 'Data points')
plt.plot(X_test_log_dis, y_predict_log_dis, color='royalblue', linewidth = 3, linestyle= '-',label ='Regression Line')

plt.title("Linear Regression of log Views versus log Disikes")
plt.xlabel("Likes")
plt.ylabel("Views")
plt.legend()
plt.show()

# The coefficients
print('Dislikes coefficient:' + '\033[1m' + '{:.2}''\033[0m'.format(coefficient_log_dis))

# The mean squared error
print('Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(model_mse_log_dis))

# The mean squared error
print('Root Mean squared error: ' + '\033[1m' + '{:.4}''\033[0m'.format(math.sqrt(model_mse_log_dis)))

# Explained variance score: 1 is perfect prediction
print('R2 score: '+ '\033[1m' + '{:.2}''\033[0m'.format(r2_score(Y_test_log_dis,y_predict_log_dis)))


# In[7]:


#plot scatter plot of views versus
#Scatter plot of the log of views versus the log of titleLength
X = titleLengths_array
Y = views_array
plt.plot(X, Y, 'o', color = 'cadetblue', label = 'ViewsVersus Title Length')
plt.title('Log Views versus log title length')
plt.xlabel('Title Length')
plt.ylabel('Views')
plt.legend()
plt.show()

meandislikes = np.mean(dislikes_array)
standard_deviation = np.std(dislikes_array)
distance_from_mean = abs(dislikes_array - meandislikes)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
dislikes_array = dislikes_array[not_outlier]

