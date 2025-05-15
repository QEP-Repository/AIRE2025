import numpy as np

def doesx1Domx2(x1,x2):
# Returns true if first solution dominates second, false otherwise. Here,
# dominance means if one solution beats the other on 1 criterion and is no
# worse on the other criterion (not strict dominance).

    result = False;
    
    if (x1[0] < x2[0] and x1[1] <= x2[1]) or (x1[0] <= x2[0] and x1[1] < x2[1]):
        result = True;
        
    return result

def ndfsort_rank1(x): 
# NDFSORT_RANK1 Fast non dominated sorting algorithm for 2 objectives only - returns only rank 1 solutions.
# 
#    XRANK = NDFSORT_RANK1(X) performed on a (N x 2) matrix X where N is the
#    number of solutions and there are 2 objectives separates out the
#    solutions into different levels (ranks) of non-domination. The returned
#    vector XRANK contains only rank 1 solutions. Solutions with rank 1 come
#    from the non-dominated front of X.
# 
#    Remarks: 
# 
#    Based on the sorting method described on page 184 of: "A fast and
#    elitist multiobjective genetic algorithm: NSGA-II" by Kalyanmoy Deb,
#    Amrit Pratap, Sameer Agarwal, T. Meyarivan. IEEE Transactions on
#    Evolutionary Computation Vol. 6, No. 2, April 2002, pp. 182-197.

    P = x.shape[0];
    npar = np.zeros((P,1)); #current domination level
    xrank = npar;  #rank vector
    
    for p in range(P):
        
        for q in range(P):
            
            if q != p:
                if doesx1Domx2(x[q,:],x[p,:]):
                    npar[p] = npar[p] + 1;
        
        if npar[p] == 0:
            xrank[p] = 1;

    return xrank