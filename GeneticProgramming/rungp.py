import numpy as np
import functions.helpers as h
from functions.init_build import init_build
from functions.fitness import *
from functions.buildNextPop import popbuild
from functions.updatestats import updatestats
from functions.displaystats import displaystats
from functions.gpinit import gpinit
from functions.mergegp import mergegp
from configs.config_Nusselt import config

def rungp():
    
    gp = config()
    
    for run in range(gp['runcontrol']['runs']):
        
        if run > 1:
            gp = config()
        
        gp = gpinit(gp)
        
        gp = h.prepare_userdata(gp)
        
        gp['state']['run'] = run+1
        
        for i in range(gp['runcontrol']['num_gen']+1):
            
            gp['state']['count'] = i
            
            if i == 0:
                gp = init_build(gp) 
            else:
                gp = popbuild(gp)
            
            gp = fitness(gp)
            
            gp = updatestats(gp)
            
            gp = displaystats(gp)
        
        if gp['state']['run'] == 1:
            gpout = gp
        else:
            gpout = mergegp(gpout, gp)
                    
        
    return gpout
