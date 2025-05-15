import functions.helpers as h
from functions.fitness import *
import numpy as np
def gppretty(gp, ID, history=False):
    
    if ID == 'best':
        win_x = gp['state']['best']['individual']
    elif isinstance(ID, int):
        if history == False:
            win_x = gp['pop'][ID]
        else:
            win_x = gp['history']['best']['individual'][ID]
    
    # disp the found variables
    h.disp_dimensionless_vars(win_x,gp['userdata']['predictor_names'])
    
    # -----------------------------------------------------------------------------
    pi_train = np.exp(np.log(gp['userdata']['Xtrain']).dot(win_x))
    pi_test = np.exp(np.log(gp['userdata']['Xtest']).dot(win_x))
    
    if gp['userdata']['add_var_to_fit'] != None:
        pi_train = np.hstack((pi_train, gp['userdata']['var2add_train'].reshape(-1,1)))
        pi_test = np.hstack((pi_test, gp['userdata']['var2add_test'].reshape(-1,1)))
        
    complexity = get_complexity(win_x)
    ff, mse, Rq, aic, bic, ytest_pred, coeffs = evalfit(gp, pi_train, pi_test, complexity)
    if gp['model']['model_type'] == 'power-law':
        h.scatter(ytest_pred, np.log(gp['userdata']['ytest']), Rq, mse)
    else:
        h.scatter(ytest_pred, gp['userdata']['ytest'], Rq, mse) 
        
    print(f'ff: {ff}')
    print(f'Rq: {Rq}')
    print(f'mse: {mse}')
    print(f'aic: {aic}')
    print(f'bic: {bic}')
    print(f'complexity: {complexity}')
    
