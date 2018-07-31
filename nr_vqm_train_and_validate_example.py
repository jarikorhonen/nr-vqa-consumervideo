# -*- coding: utf-8 -*-
"""
This script shows how to train and validate regression model to predict
MOS from the features computed with compute_features_example.m

Author: Jari Korhonen, Shenzhen University
"""

# Load libraries
import pandas
import random as rnd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np


# =======================================================================
# This function trains and validates a set 
#
def train_and_validate(seednum):
    
    # Initialization
    rnd.seed(seednum)
    
    # Load LIVE-Qualcomm features
    # For this, you need to compute them first using Matlab script
    # compute_features_example.m   
    #
    df = pandas.read_csv("./LIVE_features.csv", skiprows=[], header=None)       
    array = df.values  
    y = array[:,0]
    X = array[:,2:]  
    

    # Split data to test and validation sets randomly     
    validation_size = 0.2
    X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seednum)
    
    # Regression training here. You can use any regression model, here
    # we show the models used in the original work (SVR and RFR)
    #
    model = SVR(kernel='rbf', gamma=0.1, C=pow(2,6), epsilon=0.3)
    '''   
    model = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, 
                                  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                  max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                  min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
                                  random_state=False, verbose=0, warm_start=False)
    '''

    # Standard min-max normalization of features
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)  
    
    # Fit training set to the regression model
    model.fit(X_train, y_train)

    # Apply scaling 
    X_validation = scaler.transform(X_validation)
    
    # Predict MOS for the validation set
    y_pred = model.predict(X_validation)
    
    # Compute performance indicators
    PLCC = scipy.stats.pearsonr(y_validation,y_pred)[0]
    SRCC = scipy.stats.spearmanr(y_validation,y_pred)[0]
    RMSE = np.sqrt(mean_squared_error(y_pred,y_validation))
    
    out = [y_validation, y_pred, PLCC, SRCC, RMSE]
    
    return out
    
# ===========================================================================
# Here starts the main part of the script
#
all_scores = []
MOS_validation_all = []
MOS_predicted_all = []
PLCC_all = []
SRCC_all = []
RMSE_all = []

# Loop 100 times with different random splits
for i in range(1,101):
    new_scores = train_and_validate(i)
    MOS_validation_all.extend(new_scores[0])
    MOS_predicted_all.extend(new_scores[1])
    PLCC_all.append(new_scores[2])
    SRCC_all.append(new_scores[3])
    RMSE_all.append(new_scores[4])
    
    
# You can plot the aggregate results here if you wish
'''
plt.xlabel('Actual MOS')
plt.ylabel('Predicted MOS')
plt.title('Actual vs. Predicted MOS (LIVE-Qualcomm)')
plt.plot(MOS_validation_all,MOS_predicted_all, 'ro')
#plt.savefig('g:\HIVIQUM_LIVE-Qualcomm.png',dpi=600)
'''

# Print the average results and standard deviations
print('======================================================')
print('Average results ')
print('PLCC: ',np.mean(PLCC_all),'( std:',np.std(PLCC_all),')')
print('SRCC: ',np.mean(SRCC_all),'( std:',np.std(SRCC_all),')')
print('RMSE: ',np.mean(RMSE_all),'( std:',np.std(RMSE_all),')')
print('======================================================')
