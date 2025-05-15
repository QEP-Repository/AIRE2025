import sympy as sy
import tensorflow as tf

def config():
    
    gp = dict()
    # -------------------------------------------------------------------------
    gp['userdata'] = {}
    gp['userdata']['db_filename'] = './dataset/HDB5.2.3_STD5.xlsx'
    gp['userdata']['db_subset'] = 'STD5ELMYHITERlike'
    gp['userdata']['predictor_names'] = ['BT', 'TEV', 'AMIN', 'NEL', 'RGEO', 'KAPPAA', 'IP','MEFF', 'epsilon', 
                      'mu0', 'mp', 'e']

    #gp['userdata']['Target_name'] = 'OmegaTau'
    gp['userdata']['add_var_to_fit'] = 'DELTA'
    # -------------------------------------------------------------------------
    gp['dim_analysis'] = {}
    gp['dim_analysis']['D_matrix'] = sy.Matrix([[ 1,  1, 0,  0, 0, 0, 0, 0, -1,  1, 1, 0], 
                                                [-1, -1, 0,  0, 0, 0, 1, 0,  2, -2, 0, 1],
                                                [-2, -3, 0,  0, 0, 0, 0, 0,  4, -2, 0, 1], 
                                                [ 0,  2, 1, -3, 1, 0, 0, 0, -3,  1, 0, 0]])
    # -------------------------------------------------------------------------
    gp['runcontrol'] = {}
    gp['runcontrol']['popsize'] = 500 
    gp['runcontrol']['num_gen'] = 500
    gp['runcontrol']['runs'] = 10
    gp['runcontrol']['fitness function'] = 'bic'
    gp['runcontrol']['noise_perc_2add'] = 0
    gp['runcontrol']['modality'] = 'all_db'
    # -------------------------------------------------------------------------
    gp['selection'] = {}
    gp['selection']['tournament'] = {}
    gp['selection']['elite_fraction'] = 0.05
    gp['selection']['tournament']['p_pareto'] = 0.5
    gp['selection']['tournament']['size'] = 20
    #---------------------------------------------------------------------------
    gp['genes'] = {}
    gp['genes']['max_pi'] = 8
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
    gp['model']['model_type'] = 'power-law' # or 'custom'
    
    if gp['model']['model_type'] == 'power-law':
        gp['model']['conf_int_fit'] = True
        
    elif gp['model']['model_type'] == 'custom':
        
        # create model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        gp['model']['model'] = model
        gp['model']['epochs'] = 20
        gp['model']['optimizer'] = tf.keras.optimizers.Adam(learning_rate=0.01)
        gp['model']['loss'] = 'mse'
        gp['model']['metrics'] = None
        gp['model']['toCompile'] = True
        
    return gp