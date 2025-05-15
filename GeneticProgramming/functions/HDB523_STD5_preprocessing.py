import numpy as np
import pandas as pd
import math

def dimlessdb(db, nargout=2):
    # create dimensionless var
    rhoStar = 1.44*10**-4*(db.loc[:,'MEFF']*db.loc[:,'TEV'])**0.5/(db.loc[:,'BT']*db.loc[:,'AMIN'])
    Betat = 8.05*10**-25*db.loc[:,'NEL']*0.88*db.loc[:,'TEV']/db.loc[:,'BT']**2
    epsilon = db.loc[:,'AMIN']/db.loc[:,'RGEO']
    ka = db.loc[:,'KAPPAA']
    qcyl = 5*10**6*db.loc[:,'BT']*db.loc[:,'RGEO']*db.loc[:,'IP']**-1*epsilon**2*ka
    CulLog = 30.9 - np.log(db.loc[:,'NEL']**0.5/db.loc[:,'TEV'])
    nuStar = 5*10**-11*CulLog*db.loc[:,'NEL']*0.88*db.loc[:,'BT']*db.loc[:,'RGEO']**2*epsilon**0.5*ka*db.loc[:,'IP']**-1*db.loc[:,'TEV']**-2
    Meff = db.loc[:,'MEFF']
    IonCycfreq = db.loc[:,'BT']/db.loc[:,'MEFF']/(2*math.pi)*10
    delta1 = db.loc[:,'DELTA']+1
    R = db['RGEO']
    
    predictors = np.column_stack((rhoStar, Betat, nuStar, qcyl, delta1, ka, epsilon, Meff)).astype('float64')
    predictors = abs(predictors)
    if 'THAUTH' not in db.columns:
        return predictors, abs(IonCycfreq)
    else:
        target = np.array(db.loc[:,'THAUTH']*IonCycfreq).astype('float64')
    if nargout ==1:
        return predictors
    elif nargout==3:
        return predictors, target, abs(IonCycfreq.values)
    else:
        return predictors , target
 
def eval_weights(db):
    toks = np.unique(db['TOK'])
    a = [db['TOK'] == tok for tok in toks]
    b = [a[i].sum() for i in range(len(a))]
    nOfentries4machine = pd.DataFrame([b], columns=toks)
    print(nOfentries4machine)
    weights_values = (2 + nOfentries4machine**0.5/4)**-1
    db['WEIGHTS'] = np.zeros(db.shape[0])
    for i in range(len(toks)):
        db.loc[db['TOK'] == weights_values.columns[i],'WEIGHTS'] = weights_values.iloc[0,i]
    
    return db
    
def database_processing(db,subset):
    
    # selecting vars
    varNames = ['IP','BT','KAPPAA','AMIN','RGEO','VOL','DELTA','Q95','KAPPA',
                'WTH','MEFF','NEL','THAUTH','PHASE','TOK','TAUC93',
                'e','epsilon','mu0','mp','PLTH']
    
    db = db[varNames].copy()
    
    
    # adding epsilon and temperature to the database
    db.loc[:,'TEV'] = db['WTH']/(3*db['NEL']*0.88*db['VOL']*db['e'])
    db.loc[:,'EPSILON'] = db['AMIN']/db['RGEO']
    
    # TAUC93
    idx_tauc93 = (db['TOK'] == 'ASDEX') | (db['TOK'] == 'PDX')
    db.loc[idx_tauc93,'THAUTH']= db['THAUTH'][idx_tauc93]*db['TAUC93'][idx_tauc93]
    
    if subset == 'STD5':
        # discarding unusefull varis
        db = db.drop(['PHASE','TAUC93', 'Q95','WTH','KAPPA'], axis = 1)
        # discarding rows with nans
        db = db.dropna()
        # adding weighted scheme
        db = eval_weights(db)
        return db
    
    # SELECTING ELMY SHOTS ----------------------------------------------------
    db = db[(db['PHASE'].values != 'H') & (db['PHASE'].values != 'OHM')
            & (db['PHASE'].values != 'L') & (db['PHASE'].values != 'RI')
            & (db['PHASE'].values != 'LHLHL') & (db['PHASE'].values != 'H???')
            & (db['PHASE'].values != 'HGELM???')].copy()
    
    db = db[db['MEFF']<=3.4]
    
    if subset == 'STD5ELMYH':
        # discarding unusefull varis
        db = db.drop(['PHASE','TAUC93', 'Q95','WTH','KAPPA'], axis = 1)
        # discarding rows with nans
        db = db.dropna()
        # adding weighted scheme
        db = eval_weights(db)
        return db
    
    # SELECTING ITER LIKE SHOTS -----------------------------------------------
    iterliketoks = ['AUG','CMOD','COMPASS','D3D','JET','JFT2M','JT60U','PBXM']
    
    db= db[db['TOK'].isin(iterliketoks)]
    db = db[db['EPSILON']<0.5]
    db = db[db['Q95']>2.8]
    db = db[(db['KAPPA']<2.2) & (db['KAPPA']>1.3)]
    
    
    if subset == 'STD5ELMYHITERlike':
        # discarding unusefull varis
        db = db.drop(['PHASE','TAUC93', 'Q95','WTH','KAPPA'], axis = 1)
        # discarding rows with nans
        db = db.dropna()
        # adding weighted scheme
        db = eval_weights(db)
        return db
    else:
        print('subsets available: STD5, STD5ELMYH, STD5ELMYHITERlike')

def prepare_data_HDB23_STD5(db, config):
    
    
    if config['runcontrol']['modality'] == 'extr_jet':
        Xtrain = db[db['TOK'] != 'JET']
        Xtrain = Xtrain[Xtrain['TOK'] != 'JT60U']
        Xtest = db[db['TOK'] == 'JET']
        Xtrain = Xtrain.drop('TOK', axis=1)
        Xtest = Xtest.drop('TOK', axis=1)
        Xtrain = abs(Xtrain)
        Xtest = abs(Xtest)
    elif config['runcontrol']['modality'] == 'all_db':
        # creating training and test set
        Xtrain = db
        Xtrain = abs(Xtrain.drop('TOK', axis=1))
        # I don'r want train test division
        Xtest = Xtrain
    else:
        print('modalities: all_db/extr_jet')
        return
    
    # create dimensionless target 
    IonCycfreq_train = Xtrain.loc[:,'BT']/Xtrain.loc[:,'MEFF']/(2*math.pi)*10
    ytrain = Xtrain.loc[:,'THAUTH'].values*IonCycfreq_train.values
    
    IonCycfreq_test = Xtest.loc[:,'BT']/Xtest.loc[:,'MEFF']/(2*math.pi)*10
    ytest = Xtest.loc[:,'THAUTH'].values*IonCycfreq_test.values
    
    # exctracting weights  for the fit
    weights_train = Xtrain['WEIGHTS']**2
    weights_test = Xtest['WEIGHTS']**2
     
    # var 2 add to fit
    if config['userdata']['add_var_to_fit'] == 'DELTA':
        add = 1
    else:
        add = 0
    
    var2add_train = Xtrain.loc[:,config['userdata']['add_var_to_fit']].values+add
    var2add_test = Xtest.loc[:,config['userdata']['add_var_to_fit']].values+add
    # predictors
    columns_names = config['userdata']['predictor_names']
    Xtrain = Xtrain[columns_names]
    Xtrain = np.array(Xtrain.astype('float64'))
    Xtest = Xtest[columns_names]
    Xtest = np.array(Xtest.astype('float64'))
    
    return Xtest, ytest, weights_test, Xtrain, ytrain, weights_train, var2add_train, var2add_test