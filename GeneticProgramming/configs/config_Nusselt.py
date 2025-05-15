import sympy as sy
import tensorflow as tf
from functions.multipolyfit import multipoly
def config():
    
    gp = dict()
    # -------------------------------------------------------------------------
    gp['userdata'] = {}
    gp['userdata']['db_filename'] = './dataset/DB_Nusselt_mio.xlsx'
    gp['userdata']['predictor_names'] = ['rho','U','D','mu','cp','alpha', 'T']

    gp['userdata']['Target_name'] = 'Nu'
    gp['userdata']['add_var_to_fit'] = None
    # -------------------------------------------------------------------------
    gp['dim_analysis'] = {}
    gp['dim_analysis']['D_matrix'] = sy.Matrix([[ 1, 0, 0,  1,  0,  0, 0], 
                                                [-3, 1, 1, -1,  2,  2, 0],
                                                [ 0,-1, 0, -1, -2, -1, 0], 
                                                [ 0, 0, 0,  0, -1,  0, 1]])
    # -------------------------------------------------------------------------
    gp['runcontrol'] = {}
    gp['runcontrol']['popsize'] = 100
    gp['runcontrol']['num_gen'] = 100
    gp['runcontrol']['runs'] = 1
    gp['runcontrol']['fitness function'] = 'aic'
    gp['runcontrol']['noise_perc_2add'] = 0.1
    # -------------------------------------------------------------------------
    gp['selection'] = {}
    gp['selection']['tournament'] = {}
    gp['selection']['elite_fraction'] = 0.05
    gp['selection']['tournament']['p_pareto'] = 0.5
    gp['selection']['tournament']['size'] = 8
    #---------------------------------------------------------------------------
    gp['genes'] = {}
    gp['genes']['max_pi'] = 4
    gp['genes']['Range'] = [-2,2,1]
    gp['genes']['p_sparse'] = 1
    #--------------------------------------------------------------------------
    gp['operators'] = {}
    gp['operators']['mutation'] = {}
    gp['operators']['crossover'] = {}
    gp['operators']['direct'] = {}
    gp['operators']['mutation']['p_mutate'] = 0.49
    gp['operators']['crossover']['p_cross'] = 0.45
    gp['operators']['direct']['p_direct'] = 0.01
    
    gp['operators']['mutation']['p_mutation_list'] = [0.7, 0.1, 0.1, 0.1, 0]
    #--------------------------------------------------------------------------
    gp['model'] = {}
    gp['model']['model_type'] = 'custom' # or 'custom'
    
    if gp['model']['model_type'] == 'power-law':
        gp['model']['conf_int_fit'] = True
        
    elif gp['model']['model_type'] == 'custom':
        
        # create model
        # model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Dense(32, activation='relu'))
        # model.add(tf.keras.layers.Dense(1))
        
        # gp['model']['model'] = model
        # gp['model']['epochs'] = 20
        # gp['model']['optimizer'] = tf.keras.optimizers.Adam(learning_rate=0.01)
        # gp['model']['loss'] = 'mse'
        # gp['model']['metrics'] = None
        # gp['model']['toCompile'] = True
        
        gp['model']['model'] = multipoly(4)
        gp['model']['toCompile'] = False
        
    return gp