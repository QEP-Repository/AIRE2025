import numpy as np
import pandas as pd
from itertools import product
import math
import statsmodels.api as sm
import dill as pickle
import sympy as sy
import matplotlib.pyplot as plt
from functions.HDB523_STD5_preprocessing  import *
from sklearn.model_selection import train_test_split

def stats(var):
    print('min: '+str(np.min(var)))
    print('max: '+str(np.max(var)))
    print('mean: '+str(np.mean(var)))
    print('median: '+str(np.median(var)))
    print('std: '+str(np.std(var)))
    return None

def scatter(ypred, yactual, Rq, mse):
    fig, (ax1,ax2) = plt.subplots(2)
    fig.suptitle('R^2: '+ str(Rq) + '; mse: '+  str(mse))
    ax1.scatter(ypred,yactual)
    ax1.set_xlim([min(ax1.set_xlim()[0],ax1.set_ylim()[0]), ax1.set_xlim()[1]])
    ax1.set_ylim([min(ax1.set_xlim()[0],ax1.set_ylim()[0]), ax1.set_ylim()[1]])
    ax1.set_xlabel('$\Omega_i\cdot\;\\tau_{th}\;predicted$')
    ax1.set_ylabel('$\Omega_i\cdot\;\\tau_{th}\;actual$')
    _= ax1.plot([-100, 100], [-100,100])
    ax2.plot(ypred)
    ax2.plot(yactual)
    ax2.legend(['predicted', 'actual'])
    ax2.set_ylabel('$\Omega_i\cdot\!\\tau_{th}$')
    ax2.set_xlabel('database entries')
    # ax2.fill_between(np.arange(ypred.shape[0]), ypred_lower, ypred_upper, color='r', alpha=.7)
    return fig
   
def disp_dimensionless_vars(win_x,dimensional_var_names):
    
    for dim_var in range(win_x.shape[1]):
        sym_vars = sy.symbols(dimensional_var_names)
        i = 0
        expr = 1
        for sym_var in sym_vars:
            expr_temp = sym_var**win_x[i,dim_var]
            expr = expr*expr_temp
            i += 1
        print('pi:')
        sy.pprint(expr)
        print('')
        
    return None
    
def extrapolation(db,model,win_x,modality):
    Pi = np.exp(np.log(db).dot(win_x))
    Pi_log = np.log(Pi)
    
    model = model.get_prediction(sm.add_constant(Pi_log, has_constant='add'))
    omega_pred = model.predicted_mean
    
    ioncycfreq = db.loc[:,'BT']/db.loc[:,'MEFF']/(2*math.pi)*10
    if modality == 'tau':
        # ITER extrapolation times
        tau_pred = np.exp(omega_pred)/ioncycfreq
        tau_upper = np.exp(model.conf_int()[:,0])/ioncycfreq
        tau_lower = np.exp(model.conf_int()[:,1])/ioncycfreq
        return tau_pred, tau_upper, tau_lower
    elif modality == 'log(omega*tau)':
        omega_upper = model.conf_int()[:,0]
        omega_lower = model.conf_int()[:,1]
        return omega_pred, omega_upper, omega_lower
        
def add_noise_to_db(db, perc):
    dbnoised = db + np.random.normal(0,perc*db)
    return dbnoised

def remove_multiples(X, remove_inverse=0):
    Toremove = list()
    norms = np.sum(X**2, axis=0)
    sorted_idx = np.argsort(norms)
    X = X[:,sorted_idx]
    X = np.delete(X,0, 1)
    for i in range(X.shape[1]):
        vec1= X[:,i]
        norm_vec1 = np.sum(vec1**2)**0.5
        for j in range(i,X.shape[1],1):
            if i == j:
                continue
            vec2 = X[:,j]
            norm_vec2 = np.sum(vec2**2)**0.5
            scalar_prod = abs(vec1.dot(vec2)/(norm_vec1*norm_vec2))
            
            if scalar_prod >= 1 and ((vec1 != -vec2).all() | remove_inverse == 1):
                Toremove.append(j)
    
    if len(Toremove) == 0:
        return X
    else:
        Toremove = np.unique(Toremove)
        X = np.delete(X,Toremove,1)
        return X
    
    return X

def prepare_userdata(gp):
    db = pd.read_excel(gp['userdata']['db_filename'], header=0)
    
    if gp['userdata']['db_filename'] == './dataset/HDB5.2.3_STD5.xlsx':
        db = database_processing(db,gp['userdata']['db_subset'])
        Xtest, ytest, weights_test, Xtrain, ytrain, weights_train, var2add_train, var2add_test = prepare_data_HDB23_STD5(db, gp)
    else:
        try:
            db = add_noise_to_db(db, gp['runcontrol']['noise_perc_2add'])
        except:
                pass
        # xtrain and xtest
        Xtrain, Xtest = train_test_split(db,test_size=0.25)
        # Xtrain = db
        # Xtest = Xtrain
        # create dimensionless target 
        ytrain = Xtrain.loc[:,gp['userdata']['Target_name']].values
        ytest = Xtest.loc[:,gp['userdata']['Target_name']].values
        
        if gp['userdata']['add_var_to_fit'] != None:
            var2add_train = Xtrain.loc[:,gp['userdata']['add_var_to_fit']].values
            var2add_test = Xtest.loc[:,gp['userdata']['add_var_to_fit']].values

        else:
            var2add_train = None
            var2add_test = None
            
            
        # target normalization 
        if gp['model']['model_type'] == 'custom':
            ytrain = ytrain/np.max(ytrain)
            ytest = ytest/np.max(ytest)
            
        # extract predictors form db
        columns_names = gp['userdata']['predictor_names']
        Xtrain = Xtrain[columns_names]
        Xtrain = np.array(Xtrain.astype('float64'))
        Xtest = Xtest[columns_names]
        Xtest = np.array(Xtest.astype('float64'))
        # weights
        weights_train = np.ones(Xtrain.shape[0])
        
    # write in main dictionaries
    gp['userdata']['Xtrain'] = Xtrain
    gp['userdata']['Xtest'] = Xtest
    gp['userdata']['weights_train'] = weights_train
    gp['userdata']['ytrain'] = ytrain
    gp['userdata']['ytest'] = ytest
    gp['userdata']['var2add_train'] = var2add_train
    gp['userdata']['var2add_test'] = var2add_test


    return gp