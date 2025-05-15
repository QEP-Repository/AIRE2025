import numpy as np
from functions.ndfsort_rank1 import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import mplcursors
import functions.helpers as h
from functions.ndfsort_rank1 import ndfsort_rank1
import numpy as np
      
def popbrowser(gp, history=False):
    
    def show_annotation(sel):
        xi, yi = sel.target
        idx = int(np.argwhere(fitness == yi)[0])
        indiv = pop[idx]
        sel.annotation.set_text(f'complexity: {xi}\nfitness:{yi}\nidx: {idx}\npop: {indiv}')
    
    if history == True:
        fitness = np.array(gp['history']['best']['fitness'])
        complexity = np.array(gp['history']['best']['complexity'])
        pop = gp['history']['best']['individual']
    else:
        fitness = gp['fitness']['fitness']
        complexity = gp['fitness']['complexity']
        pop = gp['pop']
        
    x = np.vstack((fitness, complexity)).T
    xrank = ndfsort_rank1(x).reshape(-1,)
    
    pareto_indiv = [pop[i] for i in range(len(pop)) if xrank[i]==1]
    
    fig = plt.figure()
    plt.scatter(complexity, fitness)
    plt.scatter(complexity[xrank==1],fitness[xrank==1])
    plt.grid()

    cursor = mplcursors.cursor(fig, hover=True)
    cursor.connect('add', show_annotation)


