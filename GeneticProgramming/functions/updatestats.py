import numpy as np
def updatestats(gp):
    
    bestFitness = np.nanmin(gp['fitness']['fitness'])
    bestFitness_ind = np.nanargmin(gp['fitness']['fitness'])
    bestComplexity = gp['fitness']['complexity'][bestFitness_ind]
    bestInd = gp['pop'][bestFitness_ind].copy()
    bestMse = gp['fitness']['mse'][bestFitness_ind]
    bestRq = gp['fitness']['Rq'][bestFitness_ind]
    bestAic = gp['fitness']['aic'][bestFitness_ind]
    bestBic = gp['fitness']['bic'][bestFitness_ind]
    
    gp['history']['best']['fitness'].append(bestFitness)
    gp['history']['best']['complexity'].append(bestComplexity)
    gp['history']['best']['individual'].append(bestInd)
    gp['history']['best']['mse'].append(bestMse)
    gp['history']['best']['Rq'].append(bestRq)
    gp['history']['best']['aic'].append(bestAic)
    gp['history']['best']['bic'].append(bestBic)
    
    if gp['state']['count'] == 0:
        
        gp['state']['best']['fitness'] = bestFitness
        gp['state']['best']['complexity'] = bestComplexity
        gp['state']['best']['individual'] = bestInd
        gp['state']['best']['mse'] = bestMse
        gp['state']['best']['Rq'] = bestRq
        gp['state']['best']['aic'] = bestAic
        gp['state']['best']['bic'] = bestBic
    
    elif bestFitness < gp['state']['best']['fitness']:
        
        gp['state']['best']['fitness'] = bestFitness
        gp['state']['best']['complexity'] = bestComplexity
        gp['state']['best']['individual'] = bestInd
        gp['state']['best']['mse'] = bestMse
        gp['state']['best']['Rq'] = bestRq
        gp['state']['best']['aic'] = bestAic
        gp['state']['best']['bic'] = bestBic
        
    return gp
    
    
    
    