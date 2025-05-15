# import sympy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from functions.helpers import *
from functions.HDB523_STD5_preprocessing import *

if __name__ == '__main__':
    
    # laod db
    db = pd.read_excel('../dataset/HDB5.2.3_STD5.xlsx',
                       sheet_name = 'Sheet1',
                       header=0)
    
    dbil = database_processing(db,'STD5ELMYHITERlike')
    
    # Train and Test sets
    Xtrain = dbil[dbil['TOK'] != 'JET']
    Xtrain = Xtrain[Xtrain['TOK'] != 'JT60U']
    Xtest = dbil[dbil['TOK']=='JET']
    Xtrain = Xtrain.drop(['TOK', 'VOL'], axis=1)
    Xtest = Xtest.drop(['TOK', 'VOL'], axis=1)

    # evaluate dimensionless variables
    pi_train, omega_train = dimlessdb(abs(Xtrain))
    pi_test, omega_test = dimlessdb(abs(Xtest))
    log_pi_train = np.log(pi_train)
    log_omega_train = np.log(omega_train)
    log_pi_test = np.log(pi_test)
    log_omega_test = np.log(omega_test)
    
    # add constant for the intercept
    log_pi_train = sm.add_constant(log_pi_train)
    log_pi_test = sm.add_constant(log_pi_test)
    
    # fit model
    weights = (Xtrain['WEIGHTS'])**2
    omega_model = sm.WLS(log_omega_train, log_pi_train, weights=weights).fit()
    log_omega_pred_test = omega_model.predict(log_pi_test)
    mse = np.mean((log_omega_test-log_omega_pred_test)**2)
    Rq = 1-mse/np.var(log_omega_test)
    scatter(log_omega_test, log_omega_pred_test, Rq, mse)
    
    plt.figure()
    plt.title('Fit with traditional variables')
    plt.hist(log_omega_test-log_omega_pred_test)
    plt.xlabel('residuals')
    plt.ylabel('counts')
        