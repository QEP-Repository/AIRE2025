import numpy as np
import math
from functions.init_build import gene_gen
import numpy.ma as ma

def selection(gp):

    tour_ind =  np.floor( np.random.rand(gp['selection']['tournament']['size'],1)*gp['runcontrol']['popsize']).astype(int).reshape(-1,)
    
    tour_fitness = gp['fitness']['fitness'][tour_ind]
    
    bestFitness = np.nanmin(tour_fitness)
    
    bestFitness_tour_ind = tour_ind[tour_fitness == bestFitness]
    
    return bestFitness_tour_ind[0]

def mutate(gp, parent):
    

    p_mut_1 = gp['operators']['mutation']['p_mutation_list'][0]
    p_mut_2 = gp['operators']['mutation']['p_mutation_list'][1] + p_mut_1
    p_mut_3 = gp['operators']['mutation']['p_mutation_list'][2] + p_mut_2
    p_mut_4 = gp['operators']['mutation']['p_mutation_list'][3] + p_mut_3
    p_mut_5 = gp['operators']['mutation']['p_mutation_list'][4] + p_mut_4
    
    while True:
        p = np.random.random()
        
        if p < p_mut_1:
            mutationType = 1
        elif p > p_mut_1 and p < p_mut_2:
            mutationType = 2
        elif p > p_mut_2 and p < p_mut_3:
            mutationType = 3
        elif p > p_mut_3 and p < p_mut_4:
            mutationType = 4
        elif p > p_mut_4 and p < p_mut_5:
            mutationType = 5
        else:
            mutationType = 6
            
        if mutationType == 1:
            # substitude a colum with a new one
            num2remove = math.ceil( np.random.random()*parent.shape[1] )
            
            num2add = math.ceil( np.random.random()*parent.shape[1] )
            
            indPi2remove = np.random.permutation(parent.shape[1])[:num2remove]
            
            new_pi = gene_gen(gp, n_pi=num2add)
            
            mutated_gene = np.concatenate((parent, new_pi), axis=1)
            
            mutated_gene = np.delete(mutated_gene,indPi2remove, axis=1) 
            
        elif mutationType == 2:
            # remove a column 
            num2remove = math.ceil( np.random.random()*parent.shape[1] )
            
            indPi2remove = np.random.permutation(parent.shape[1])[:num2remove]
            
            mutated_gene = np.delete(parent,indPi2remove, axis=1) 
            
        elif mutationType == 3:
            # sum or subtracts a number of columns togheter
            if parent.shape[1] == 1:
                
                mutation_bool = False
                
                continue
            
            num2sum = math.ceil( np.random.random()*(parent.shape[1]-1))+1
            
            indPi2sum = np.random.permutation(parent.shape[1])[:num2sum]
            
            sign = 2*np.round(np.random.rand(num2sum,1))-1
            
            new_pi = parent[:,indPi2sum].dot(sign).reshape(-1,1)
            
            mutated_gene = np.concatenate((parent, new_pi), axis=1)
            
            mutated_gene = np.delete(mutated_gene,indPi2sum, axis=1) 
        
        elif mutationType == 4:
            # take a column and sum or subtracting to other columns
            if parent.shape[1] == 1:
                
                mutation_bool = False
                
                continue
            
            num2sum = math.ceil( np.random.random()*(parent.shape[1]-1))+1
            
            indPi2sum = np.random.permutation(parent.shape[1])[:num2sum]
            
            for i in range(num2sum-1):
                
                sign = 2*np.round(np.random.rand(2,1))-1
                
                idx = [indPi2sum[0], indPi2sum[i+1]]
                
                new_pi = parent[:,idx].dot(sign).reshape(-1,1)
                
                mutated_gene = np.concatenate((parent, new_pi), axis=1)
            
            mutated_gene = np.delete(mutated_gene,indPi2sum, axis=1)
            
        elif mutationType == 5:
            
            if parent.shape[1] == 1:
                
                mutation_bool = False
                
                continue
            
            D_matrix = np.array(gp['dim_analysis']['D_matrix']).astype('int')
            
            idxs_already_dimless = np.argwhere((np.sum(D_matrix, axis=0) == 0))
            
            if idxs_already_dimless.shape[0] == 0:
                
                mutation_bool = False
                
                continue
            idx_already_dimless = np.random.permutation(idxs_already_dimless)[0]
            
            num2put2zero = math.ceil( np.random.random()*(parent.shape[1]-1))+1
            
            indPi2mutate = np.random.permutation(parent.shape[1])[:num2put2zero]
            
            parent[idx_already_dimless,indPi2mutate] = 0
            
            mutated_gene = parent
            
        elif mutationType == 6:
            
            while True:
                var2change = math.floor(np.random.random()*(parent.shape[0]))
                
                pi2change = math.floor(np.random.random()*(parent.shape[1]))
                
                if parent[var2change, pi2change] == 0:
                    continue
                else:
                
                    D_matrix = np.array(gp['dim_analysis']['D_matrix']).astype('int')
                    
                    idxs_already_dimless = np.argwhere((np.sum(D_matrix, axis=0) == 0)).reshape(-1,)
                    
                    idxs2remove = np.append(idxs_already_dimless, var2change)
                    
                    idxs2permutate = np.delete(np.arange(0,D_matrix.shape[1]),idxs2remove)
                    
                    dim2match = D_matrix[:,var2change]*parent[var2change, pi2change]
                    
                    indD = np.random.permutation(idxs2permutate)[:D_matrix.shape[0]]
                    
                    try: 
                        x = np.linalg.solve(D_matrix[:,indD],dim2match)
                        mx = ma.masked_array(x, mask=x==0)
                        x = (x/np.min(np.absolute(mx))).astype('int')
                    except:
                        continue
                    
                    parent[var2change, pi2change] = 0
        
                    parent[indD, pi2change] = parent[indD, pi2change]+x
                    
                    mutated_gene = parent
                    
                    break
            
            
        mutation_bool = True
            
        if mutation_bool == True:
            break
            
    
    zeros_idx = np.argwhere(np.sum(abs(mutated_gene), axis = 0) == 0)
    mutated_gene = np.delete(mutated_gene, zeros_idx, axis=1)
    
    return mutated_gene              

def crossover(gp, mum, dad):
    
    min_pi = np.min([mum.shape[1], dad.shape[1]])
    
    num2switch = math.ceil( np.random.random()*min_pi )
    
    dadIndPi2switch = np.random.permutation(dad.shape[1])[:num2switch]
    mumIndPi2switch = np.random.permutation(mum.shape[1])[:num2switch]

    dad_pi = dad[:,dadIndPi2switch]
    mum_pi = mum[:,mumIndPi2switch]
    
    if num2switch == 1:
        dad_pi = dad_pi.reshape(-1,1)
        mum_pi = mum_pi.reshape(-1,1)
    
    son = np.concatenate((dad, mum_pi), axis=1)
    son = np.delete(son, dadIndPi2switch, axis=1)
    
    daughter = np.concatenate((mum, dad_pi), axis=1)
    daughter = np.delete(daughter, mumIndPi2switch, axis=1)
    
    zeros_idx = np.argwhere(np.sum(abs(son), axis = 0) == 0)
    son = np.delete(son, zeros_idx, axis=1)
    zeros_idx = np.argwhere(np.sum(abs(daughter), axis = 0) == 0)
    daughter = np.delete(daughter, zeros_idx, axis=1)
    
    return son, daughter
    
def popbuild(gp):
    
    p_mutate = gp['operators']['mutation']['p_mutate']
    p_cross = gp['operators']['crossover']['p_cross'] 
    p_direct = gp['operators']['direct']['p_direct']
    max_pi =   gp['genes']['max_pi']
    
    num2build =  math.floor( (1 - gp['selection']['elite_fraction']) * gp['runcontrol']['popsize']);
    num2skim = gp['runcontrol']['popsize'] - num2build
    
    new_pop = list()
    while True:
        
        p = np.random.random()
        
        if p < p_mutate: # select mutation
            eventType = 1
        elif p < p_mutate + p_direct: # select direct reproductio
            eventType = 2
        else: # select crossover
            eventType = 3
            
        
        if eventType == 1:
            
            for j in range(5):
                
                parentIndex = selection(gp)
                parent = gp['pop'][parentIndex]
                
                mutated_gene = mutate(gp,parent)
                
                if mutated_gene.shape[1] <= max_pi:
                    
                    if len(new_pop) < num2build:
                        new_pop.append(mutated_gene)
                        break
        
        elif eventType == 2:
            
            parentIndex = selection(gp)
            parent = gp['pop'][parentIndex]
            
            if len(new_pop) < num2build:
                new_pop.append(parent)
        
        else: 
            
            for j in range(5):
                parentIndex = selection(gp)
                dad = gp['pop'][parentIndex]
                
                parentIndex = selection(gp)
                mum = gp['pop'][parentIndex]
                
                son, daughter = crossover(gp, mum, dad)
                
                if son.shape[1] <= max_pi and daughter.shape[1] <= max_pi:
                    
                    if len(new_pop) < num2build-1:
                        new_pop.append(son)
                        new_pop.append(daughter)
                    break
                
            if len(new_pop) == num2build:
                break
                
    
    sortedIndex2skim = np.argsort(gp['fitness']['fitness'])[:num2skim].astype('int')
    elite = [gp['pop'][i] for i in sortedIndex2skim]
    new_pop = new_pop + elite
    
    gp['pop'] = new_pop
    
    return gp
