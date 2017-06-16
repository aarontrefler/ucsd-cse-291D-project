import random, pylab, math
import collections
import numpy
from sklearn.linear_model import LogisticRegression
from preprocessing import *

def hard_EM(train, test, gender='m',\
			init_lr=1e-2, tau=10**1.73, N_iter=2, eps=1e-1, max_perf_inc=1.04, lr_dec=0.95):
	"""
	Args:
		train: train set data, a list of tuples (person, info)
		test: test set data, a list of tuples (person, info)
		gender: 'm' or 'w', indicating gender of the decision makers
		init_lr: initial learning rate
		tau: reciprocal variance of prior, serves as regularization coefficient
		N_iter: number of iterations
		eps: termination criterion for E-step
	"""
	(IDX_OF_RATER, IDX_OF_RATEE) = (0, 1) if gender == 'm' else (1, 0)
	IDX_OF_INFO = 2

	feature = lambda x: numpy.array([float(x[IDX_OF_INFO][gender+a]) for a in attrs])
	decision = lambda x: x[IDX_OF_INFO][gender+'dec']
	rater = lambda x: x[IDX_OF_RATER]

	#############################################################################################
	# do preprocessing                                                                          #
	#############################################################################################
	X_train, X_test = map(feature, train), 	map(feature, test)
	Y_train, Y_test = map(decision, train), map(decision, test)
	P_train, P_test = map(rater, train),	map(rater, test)

	LEN_FEATURES = len(X_train[0])
	Raters = set(d[IDX_OF_RATER] for d in train)
	Ratees = set(d[IDX_OF_RATEE] for d in train)

	W = {p:numpy.random.uniform(low=-1,high=1,size=LEN_FEATURES) for p in Raters}
	b = {p:0. for p in Raters}

	tau_b, tau_W = tau, tau * numpy.ones(shape=LEN_FEATURES)
	model = LogisticRegression()
	model.fit(X_train, Y_train)
	# mu_b,  mu_W  = -5.03604141, numpy.array([ 0.67764858, -0.16627213,  0.24595321, -0.16821214, -0.03345339, 0.27498794])
	mu_b,  mu_W  = model.intercept_[0], model.coef_[0]

	#############################################################################################
	# do training                                                                               #
	#############################################################################################

	def log_prob(W, b):
		C = 0
		C -= sum(0.5 * tau_b * (b[p]-mu_b) ** 2 for p in Raters)
		C -= sum(0.5 * sum(tau_W * (W[p]-mu_W) ** 2) for p in Raters)
		for x, y, p in zip(X_train, Y_train, P_train):
			theta = numpy.dot(W[p], x) + b[p]
			C -= numpy.log(1 + numpy.exp(-theta)) if y else numpy.log(1 + numpy.exp(theta))
		return C

	scores_train = [evaluate(X_train, Y_train, P_train, W, b)]
	scores_test  = [evaluate(X_test, Y_test, P_test, W, b)]
	print scores_train[-1], scores_test[-1]

	for i in range(N_iter):
		print "Iteration ", i
		
		# hard E step - estimate latent variables by gradient descent
		new_log = log_prob(W, b)
		lr = init_lr
		while True:
			old_log = new_log
			# gradient w.r.t. log-pirors
			p_b = {p:tau_b * (mu_b-b[p]) for p in b}
			p_W = {p:tau_W * (mu_W-W[p]) for p in W}

			# gradient w.r.t. log-likelihoods
			l_b = {p:0. for p in b}
			l_W = {p:numpy.zeros(shape=LEN_FEATURES) for p in W}
			for x, y, p in zip(X_train, Y_train, P_train):
				theta = numpy.dot(W[p], x) + b[p]
				e = 1. / (1 + numpy.exp(theta)) if y else - 1. / (1 + numpy.exp(-theta))
				l_b[p] += e
				l_W[p] += e * x

			# update latent variables
			b_new = {p:b[p] + lr * (l_b[p] + p_b[p]) for p in Raters}
			W_new = {p:W[p] + lr * (l_W[p] + p_W[p]) for p in Raters}
			new_log = log_prob(W_new, b_new)
			
			# termination judge
			if abs(old_log - new_log) < eps: 
				W, b = W_new, b_new
				break
			if new_log < old_log:
				new_log = old_log
				lr *= lr_dec
			else:
				W, b = W_new, b_new

		print '\t', lr, '\t', new_log

		# M step - estimate hyper-parameters
		mu_b = numpy.mean(b.values())
		mu_W = numpy.mean(W.values(),axis=0)

		# do evaluation
		scores_train += evaluate(X_train, Y_train, P_train, W, b),
		scores_test += evaluate(X_test, Y_test, P_test, W, b),
		print scores_train[-1], scores_test[-1]

	# pylab.plot(range(N_iter+1), scores_t)
	# pylab.plot(range(N_iter+1), scores_v)
	return mu_W, mu_b, W, b

def evaluate(X, Y, P, W, b, mu_W=None, mu_b=None):
	"""
	Evaluate the accuracy of model
	"""
	if mu_b is None: mu_b = numpy.mean(b.values())
	if mu_W is None: mu_W = numpy.mean(W.values(),axis=0)
	Y_pred = []
	for x, y, p in zip(X, Y, P):
		W_p = W[p] if p in W else mu_W
		b_p = b[p] if p in b else mu_b
		theta = numpy.dot(W_p, x) + b_p
		Y_pred += 1 if theta >= 0 else 0,

	return accuracy(Y_pred, Y)


trainM, testM, trainW, testW = readData()

# Matrices = cPickle.load(open("mf_trainM.pkl","rb"))

mu_W, mu_b, W, b = hard_EM(trainM, testM)
