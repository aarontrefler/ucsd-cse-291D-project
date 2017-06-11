import collections
import random
import pylab
import numpy

from preprocessing import *


def collaborative_filtering(train, test, aspect, gender='m', lam_beta=2, lam_gamma=6, Niter=10, niter=7, k=3):
	# preprocessing
	R = {}
	A = collections.defaultdict(set) # a for rater
	B = collections.defaultdict(set) # b for ratee

	pair = lambda (m,w): (m,w) if gender == 'm' else (w,m)
	score = lambda d: d[gender+aspect]
	residuals = lambda : {(a,b): R[a,b] - alpha - betaA[a] - betaB[b] - numpy.dot(gammaA[a], gammaB[b]) for a, b in R}

	for p, d in train:
		a, b = pair(p)
		if gender+aspect in d:
			A[b].add(a)
			B[a].add(b)
			R[a,b] = score(d)

	X_train, X_test = [pair(p) for p,_ in train], [pair(p) for p,_, in test]
	Y_train, Y_test = [score(d) for _,d in train],  [score(d) for _,d in test]

	alpha = mean(Y_train)
	betaA = collections.defaultdict(float)
	betaB = collections.defaultdict(float)
	gammaA = collections.defaultdict(lambda: numpy.random.uniform(low=-0.1,high=0.1,size=k))
	gammaB = collections.defaultdict(lambda: numpy.random.uniform(low=-0.1,high=0.1,size=k))

	# alternating least square
	MSE_train = [MSE(Y_train, [alpha] * len(Y_train))]
	MSE_test  = [MSE(Y_test, [mean(Y_test)] * len(Y_test))]
	
	for p in range(2 * Niter):
		for _ in range(niter):
			# update alpha
			res = residuals()
			alpha += sum(res.values()) * 1. / len(res)

			# update beta_a
			res = residuals()
			for a in B:
				betaA[a] = sum(res[a,b] + betaA[a] for b in B[a]) * 1. / (lam_beta + len(B[a]))
			
			# update beta_b
			res = residuals()
			for b in A:
				betaB[b] = sum(res[a,b] + betaB[b] for a in A[b]) * 1. / (lam_beta + len(A[b]))
			
			res = residuals()
			if p % 2:
			# update gamma_b
				for b in A:
					for j in range(k):
						gammaB[b][j] = sum((res[a,b] + gammaA[a][j]*gammaB[b][j]) * gammaA[a][j] for a in A[b]) * 1.\
							 / (lam_gamma + sum(gammaA[a][j]**2 for a in A[b]))
				# compute MSE
			else:
			# update gamma_a
				for a in B:
					for j in range(k):
						gammaA[a][j] = sum((res[a,b] + gammaA[a][j]*gammaB[b][j]) * gammaB[b][j] for b in B[a]) * 1.\
							 / (lam_gamma + sum(gammaB[b][j]**2 for b in B[a]))

		# compute MSE
		MSE_train += evaluate(X_train, Y_train, alpha, betaA, betaB, gammaA, gammaB),
		MSE_test += evaluate(X_test, Y_test, alpha, betaA, betaB, gammaA, gammaB),

	pylab.plot(range(len(MSE_train)), MSE_train, '-b')
	pylab.plot(range(len(MSE_test)), MSE_test, '-g')
	pylab.show()
	print MSE_train[-1], MSE_test[-1]
	return alpha, betaA, betaB, gammaA, gammaB

def evaluate(X, Y, alpha, betaA, betaB, gammaA, gammaB):
	mu_beta_A = mean(betaA.values())
	mu_beta_B = mean(betaB.values())
	mu_gamma_A = mean(gammaA.values())
	mu_gamma_B = mean(gammaB.values())

	Y_pred = []
	for (a, b), y in zip(X,Y):
		bA = betaA.get(a, mu_beta_A)
		bB = betaB.get(b, mu_beta_B)
		gA = gammaA.get(a, mu_gamma_A)
		gB = gammaB.get(b, mu_gamma_B)
		Y_pred += alpha + bA + bB + numpy.dot(gA, gB),

	return MSE(Y_pred, Y)

data = [(x, Ratings[x]) for x in Ratings if 'mattr' in  Ratings[x]]
random.shuffle(data)
train, test = data[:len(data)*9/10], data[len(data)*9/10:]

alpha, betaA, betaB, gammaA, gammaB = collaborative_filtering(train, test, 'attr')