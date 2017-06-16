# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
# Altered by Aaron Trefler (2016)

import numpy as np

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, verbose=0, output_dir='.'):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if ~np.isnan(R[i][j]):
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        
        #eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if ~np.isnan(R[i][j]):
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        
        if (verbose > 0):
            print("step:", step, "error", e)
        
        if (step % 10 == 0):
            np.savetxt((output_dir + "/P.txt"), P, delimiter=',')
            np.savetxt((output_dier + "/Q.txt"), Q.T, delimiter=',')
            fh = open((output_dir + "/error.txt", "a+")
            fh.write(str(step))
            fh.write(", ")
            fh.write(str(e))
            fh.write("\n")
            fh.close()
        
        if e < 0.001:
            break
    return P, Q.T