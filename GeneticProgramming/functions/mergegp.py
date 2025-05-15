import numpy as np
def mergegp(gp1, gp2):
    
    gp1['pop'] = gp1['pop'] + gp2['pop']
    gp1['runcontrol']['popsize'] = gp1['runcontrol']['popsize']+gp2['runcontrol']['popsize']
    
    gp1['fitness']['fitness'] = np.hstack((gp1['fitness']['fitness'], gp2['fitness']['fitness']))
    gp1['fitness']['aic'] = np.hstack((gp1['fitness']['aic'], gp2['fitness']['aic']))
    gp1['fitness']['bic'] = np.hstack((gp1['fitness']['bic'], gp2['fitness']['bic']))
    gp1['fitness']['complexity'] = np.hstack((gp1['fitness']['complexity'], gp2['fitness']['complexity']))
    gp1['fitness']['mse'] = np.hstack((gp1['fitness']['mse'], gp2['fitness']['mse']))
    gp1['fitness']['Rq'] = np.hstack((gp1['fitness']['Rq'], gp2['fitness']['Rq']))
    
    gp1['history']['best']['fitness'] = gp1['history']['best']['fitness'] + gp2['history']['best']['fitness']
    gp1['history']['best']['aic'] = gp1['history']['best']['aic'] + gp2['history']['best']['aic']
    gp1['history']['best']['bic'] = gp1['history']['best']['bic'] + gp2['history']['best']['bic']
    gp1['history']['best']['complexity'] = gp1['history']['best']['complexity'] + gp2['history']['best']['complexity']
    gp1['history']['best']['mse'] = gp1['history']['best']['mse'] + gp2['history']['best']['mse']
    gp1['history']['best']['Rq'] = gp1['history']['best']['Rq'] + gp2['history']['best']['Rq']
    gp1['history']['best']['individual'] = gp1['history']['best']['individual'] + gp2['history']['best']['individual']
    
    if gp2['state']['best']['fitness'] < gp1['state']['best']['fitness']:
        gp1['state']['best'] = gp2['state']['best']
        
        
    return gp1