# init_build
import numpy as np

def gene_gen(gp, n_pi=None):
    nullsmatrix = gp['dim_analysis']['nullsmatrix']
    Range = gp['genes']['Range'] 
    max_pi = gp['genes']['max_pi']
    if n_pi == None:
        n_pi = np.random.randint(1,max_pi+1)
    gene = np.zeros((nullsmatrix.shape[0],1))
    for j in range(n_pi):
        coeffs = np.random.randint(Range[0]/Range[2],Range[1]/Range[2], size=nullsmatrix.shape[1])*Range[2]
        p_sparse = np.random.random()
        if p_sparse < gp['genes']['p_sparse']:
            nzeros = np.random.randint(1,len(coeffs))
            idxs_zeros = np.random.permutation(len(coeffs))[:nzeros]
            coeffs[idxs_zeros] = 0
        # matrix multiplication between matrix of basis vector andcoefficients
        # this gives me a possible solution of the problem Dx = 0
        x = np.matmul(nullsmatrix,coeffs).reshape(nullsmatrix.shape[0],1)
        gene = np.concatenate((gene,x),axis=1)
    gene = np.delete(gene,0, axis=1)
    zeros_idx = np.argwhere(np.sum(abs(gene), axis = 0) == 0)
    gene = np.delete(gene, zeros_idx, axis=1)
    return gene
    
def init_build(gp):
    
    # extracting values from 
    D = gp['dim_analysis']['D_matrix']
    pop_size = gp['runcontrol']['popsize']
    # find nullspace of the matrix and print them
    D_nullspace = D.nullspace() #sy.pprint(D_nullspace) 
    # transform in numpy format and stuck basis vectors in matrix form
    nullsarrays = list(np.array(D_nullspace).astype('float64'))
    gp['dim_analysis']['nullsmatrix'] = np.concatenate(nullsarrays, axis=1)
    pop = list()
    for i in range(pop_size):
        gene = gene_gen(gp)
        pop.append(gene)
    
    # initial population
    gp['pop'] = pop
    
    return gp