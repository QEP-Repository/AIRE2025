import sympy as sy
def config_Heat_transfer():
    
    config = dict()
    config['db_filename'] = './dataset/Dati_Heat_Transfer.xlsx'
    config['D_matrix'] = sy.Matrix([[0, 0, 1, 1], 
                                    [0, 0, -1, -1],
                                    [0, 0, -3, -3], 
                                    [2, 3, 0, 1]])
                                    
    config['alpha_interval'] = [-2, -1, 0, 1, 2]
    config['columns_names'] = ['A [m^2]','V [m^3]','h [W/m^2 K]','kappa [w/mK]']
    config['batch_size'] = 3
    config['conf_int_fit'] = False
    config['filename2save'] = None
    
    return config
