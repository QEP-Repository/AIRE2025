def gpinit(gp):
    
    gp['fitness']= {}
    gp['history'] = {}
    gp['history']['best'] = {}
    gp['state'] = {}
    gp['state']['best'] = {}
    gp['state']['count'] = 0
    gp['history']['best']['fitness'] = list()
    gp['history']['best']['complexity'] = list()
    gp['history']['best']['individual'] = list()
    gp['history']['best']['mse'] = list()
    gp['history']['best']['Rq'] = list()
    gp['history']['best']['aic'] = list()
    gp['history']['best']['bic'] = list()
    
    return gp