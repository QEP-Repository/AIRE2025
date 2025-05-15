# import sympy 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from functions.helpers import *
from functions.HDB523_STD5_preprocessing import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    db = pd.read_excel('../dataset/HDB5.2.3_STD5.xlsx',
                       sheet_name = 'Sheet1',
                       header=0)
    
    dbil = database_processing(db,'STD5ELMYHITERlike')
    dbil = dbil.drop(['TOK', 'VOL'], axis=1)

    # evaluate dimensionless variables 
    pi, omega = dimlessdb(abs(dbil))
    log_pi = np.log(pi)
    log_omega = np.log(omega)
    
    # add constant for the intercept
    log_pi = sm.add_constant(log_pi)
    
    # fit model
    weights = (dbil['WEIGHTS'])**2
    omega_model = sm.WLS(log_omega, log_pi, weights=weights).fit()
    
    log_omega_pred = omega_model.predict(log_pi)
    
    mse = np.mean((log_omega-log_omega_pred)**2)
    
    Rq = 1-mse/np.var(log_omega)
    
    scatter(log_omega, log_omega_pred, Rq, mse)
