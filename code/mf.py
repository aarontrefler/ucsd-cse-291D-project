# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
# Altered by Aaron Trefler (2016)
#
# Altered by Wangmuge Qin (2016)

import numpy as np
import collections
from preprocessing import *

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
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, verbose=0,output_dir='.'):
    
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
        
        """
        if (step % 10 == 0):
            np.savetxt((output_dir + "/P.txt"), P, delimiter=',')
            np.savetxt((output_dier + "/Q.txt"), Q.T, delimiter=',')
            fh = open((output_dir + "/error.txt", "a+")
            fh.write(str(step))
            fh.write(", ")
            fh.write(str(e))
            fh.write("\n")
            fh.close()
        """

        if e < 0.001:
            break
    return P, Q.T


def get_latent_vector(train, gender='m', K=2):
	pair = lambda m,w: (m,w) if gender == 'm' else (w,m)
	waves = collections.defaultdict(dict)
	for m, w, d in train:
		wid = int(m[0])
		m, w = int(m[1]), int(w[1])
		waves[wid][pair(m,w)] = 1 if d[gender+'dec'] == 1 else -1

	R, P, Q = {}, {}, {}
	for wid in waves:
		print "wave %d" % wid
		N_rater = int(max(a for a,b in waves[wid]))
		N_ratee = int(max(b for a,b in waves[wid]))
		print "%d by %d" % (N_rater, N_ratee)
		R[wid] = np.zeros(shape=(N_rater,N_ratee))
		P[wid] = np.random.uniform(low=-1,high=1,size=(N_rater,K))
		Q[wid] = np.random.uniform(low=-1,high=1,size=(N_ratee,K))
		for (a,b),d in waves[wid].items():
			R[wid][a-1,b-1] = d
		P[wid], Q[wid] = matrix_factorization(R[wid], P[wid], Q[wid], K)
	return R, P, Q

# R, P, Q = get_latent_vector(trainM)
