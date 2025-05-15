from sklearn.tree import DecisionTreeRegressor
import sympy as sy
import tensorflow as tf
def config():
    
    config = dict()
    
    # -------------------------------------------------------------------------
    config['userdata'] = {}
    config['userdata']['db_filename'] = './dataset/DB_Reynolds.xlsx'
    config['userdata']['predictor_names'] = ['rho','U','D','mu', 'cp','alpha', 'T']
    config['userdata']['D_matrix'] = sy.Matrix([[ 1, 0, 0,  1,  0,  0, 0], 
                                    [-3, 1, 1, -1,  2,  2, 0],
                                    [ 0,-1, 0, -1, -2, -1, 0], 
                                    [ 0, 0, 0,  0, -1,  0, 1]])
    config['userdata']['Target_name'] = 'Cd'
    
    # -------------------------------------------------------------------------
    config['runcontrol'] = {}
    config['runcontrol']['alpha_interval'] = [-1, 0, 1]
    config['runcontrol']['fitness function'] = 'mse'
    # -------------------------------------------------------------------------
    config['model'] = {}
    config['model']['model_type'] = 'custom' # or 'custom'
    
    if config['model']['model_type'] == 'power-law':
        config['model']['conf_int_fit'] = True
        
    elif config['model']['model_type'] == 'custom':
        
        # create model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        config['model']['model'] = model
        config['model']['epochs'] = 5
        config['model']['optimizer'] = tf.keras.optimizers.Adam(learning_rate=0.001)
        config['model']['loss'] = 'mse'
        config['model']['metrics'] = None
        config['model']['toCompile'] = True
        
    return config