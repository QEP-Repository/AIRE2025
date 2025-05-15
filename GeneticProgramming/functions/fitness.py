import numpy as np
from itertools import product
import statsmodels.api as sm
import functions.helpers as h
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

def evalfit_power_law(gp, pi_train, pi_test, complexity):
    
    conf_int_fit = gp['model']['conf_int_fit']
    ytrain = gp['userdata']['ytrain']
    ytest = gp['userdata']['ytest']
    weights_train = gp['userdata']['weights_train']
    
    ytrain_log = np.log(ytrain)
    ytest_log = np.log(ytest)
    pi_train_log = np.log(pi_train)
    pi_test_log = np.log(pi_test)
    
    try:
        # log fit dimless var to target 
        model = sm.WLS(ytrain_log, sm.add_constant(pi_train_log), weights=weights_train).fit()
        model_pred = model.get_prediction(sm.add_constant(pi_test_log))
        
        ytest_log_pred = model_pred.predicted_mean
        mse_mean = np.mean((ytest_log_pred - ytest_log)**2)
        Rq_mean = 1-mse_mean/np.var(ytest_log)
        
        if conf_int_fit == True:
            ytest_log_pred_upper = model_pred.conf_int()[:,0]
            mse_upper = np.mean((ytest_log_pred_upper - ytest_log)**2)
            Rq_upper =  1-mse_upper/np.var(ytest_log)
            
            ytest_log_pred_lower = model_pred.conf_int()[:,1]
            mse_lower = np.mean((ytest_log_pred_lower - ytest_log)**2)
            Rq_lower =  1-mse_lower/np.var(ytest_log)
            
            mse = (mse_mean+mse_upper+mse_lower)/3
            Rq = (Rq_mean+Rq_upper+Rq_lower)/3
        else:
            mse = mse_mean
            Rq = Rq_mean
            
        N = ytest.shape[0]
        k = complexity
        aic = N*np.log(mse) + 2*k
        bic =  N*np.log(mse) + k*np.log(N)
        coeffs = model.params
    except:
        mse = float('inf')
        Rq = 0
        aic = float('inf')
        bic = float('inf')
        coeffs = float('inf')
        ytest_log_pred = float('inf')
    
    if gp['runcontrol']['fitness function'] == 'mse':
        ff = mse
    elif gp['runcontrol']['fitness function'] == 'aic':
        ff = aic
    elif gp['runcontrol']['fitness function'] == 'bic':
        ff = bic
        
    return ff, mse, Rq, aic, bic, ytest_log_pred, coeffs

def evalfit_custom(gp, pi_train, pi_test, complexity):
    
    ytrain = gp['userdata']['ytrain']
    ytest = gp['userdata']['ytest']
    try: 
        if gp['model']['toCompile'] == True:
            pi_train = pi_train/np.max(pi_train, axis=0)
            pi_test = pi_test/np.max(pi_test, axis=0)
            model = tf.keras.models.clone_model(gp['model']['model'])
            model.compile(optimizer=gp['model']['optimizer'],
                          loss=gp['model']['loss'],
                          metrics=gp['model']['metrics'])
            history = model.fit(pi_train, ytrain,
                                epochs = gp['model']['epochs'],
                                validation_split = 0.1, 
                                verbose=1)
            history_dict = history.history
            val_loss = history_dict['val_loss'][-1]
        else:
            # pi_train = pi_train/np.max(pi_train, axis=0)
            # pi_test = pi_test/np.max(pi_test, axis=0)
            model = gp['model']['model']
            model.fit(pi_train, ytrain)
           
            
        coeffs = 0
        ytest_pred = model.predict(pi_test).reshape(ytest.shape)
        mse = np.mean((ytest_pred - ytest)**2)
        Rq = 1-mse/np.var(ytest)
        
        N = ytest.shape[0]
        aic = N*np.log(mse) + 2*complexity
        bic =  N*np.log(mse) + complexity*np.log(N)
        
    except:
            mse = float('inf')
            Rq = float('inf')
            aic = float('inf')
            bic = float('inf')
            ytest_pred = float('inf')
            coeffs = float('inf')
            val_loss = float('inf')
    
    if gp['runcontrol']['fitness function'] == 'mse':
        ff = mse
    elif gp['runcontrol']['fitness function'] == 'aic':
        ff = aic
    elif gp['runcontrol']['fitness function'] == 'bic':
        ff = bic
        
    return ff, mse, Rq, aic, bic, ytest_pred, coeffs

def evalfit(gp, pi_train, pi_test, complexity):
    
    if gp['model']['model_type'] == 'power-law':
        return evalfit_power_law(gp, pi_train, pi_test, complexity)
    elif gp['model']['model_type'] == 'custom':
        return evalfit_custom(gp, pi_train, pi_test, complexity)
    else:
        raise Exception('model_type should be: "power-law" or "custom"')
        
def fitness(gp):
    
    Xtrain = gp['userdata']['Xtrain']
    Xtest = gp['userdata']['Xtest']
    var2add_train = gp['userdata']['var2add_train']
    var2add_test = gp['userdata']['var2add_test']
    
    ff_list = list()
    complexity_list = list()
    mse_list = list()
    Rq_list = list()
    aic_list = list()
    bic_list = list()
    
    for gene in gp['pop']:
        
        pi_train = np.exp(np.log(Xtrain).dot(gene))
        pi_test = np.exp(np.log(Xtest).dot(gene))
        
        if gp['userdata']['add_var_to_fit'] != None:
            pi_train = np.hstack((pi_train, var2add_train.reshape(var2add_train.shape[0],1)))
            pi_test = np.hstack((pi_test, var2add_test.reshape(var2add_test.shape[0],1)))
            
        # complexity = np.sum(np.abs(gene))
        complexity = get_complexity(gene)
        
        if complexity == 0:
            ff = float('inf')
            mse = float('inf')
            Rq = float('inf')
            aic = float('inf')
            bic = float('inf')
            
        else:
            ff, mse, Rq, aic, bic, _, _ = evalfit(gp, pi_train, pi_test, complexity)
            
        ff_list.append(ff)
        mse_list.append(mse)
        Rq_list.append(Rq)
        aic_list.append(aic)
        bic_list.append(bic)
        complexity_list.append(complexity)
        
    gp['fitness']['fitness'] = np.array(ff_list)
    gp['fitness']['complexity'] = np.array(complexity_list)
    gp['fitness']['mse'] = np.array(mse_list)
    gp['fitness']['Rq'] = np.array(Rq_list)
    gp['fitness']['aic'] = np.array(aic_list)
    gp['fitness']['bic'] = np.array(bic_list)

    return gp


def get_complexity(gene):
    
    complexity = np.sum(np.abs(gene))
    
    return complexity