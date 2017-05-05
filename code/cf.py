import csv
import random, pylab
import collections

def readData():
	with open('..\data\Speed Dating Data.csv', 'rU') as csvfile:
		data = [r for r in csv.reader(csvfile)]
		scheme = {t: i for i, t in enumerate(data[0])}
		ischeme = {i: t for t, i in scheme.items()}
		data = [map(lambda s: int(s) if s.isdigit() else s, d) for d in data[1:]]

	people = {}
	rating = {}
	for d in data:
		wave = d[scheme['wave']]
		gender = 'm' if d[scheme['gender']] else 'w'
		pid = d[scheme['id']]
		partner = d[scheme['partner']]

		# personal data
		if wave not in people: people[wave] = {}
		if gender not in people[wave]: people[wave][gender] = {}
		if pid not in people[wave][gender]:
			people[wave][gender][pid] = {}
			p = people[wave][gender][pid]
			for i in range(scheme['age'], scheme['amb5_1']+1):
				if d[i] != "": p[ischeme[i]] = d[i]
			for i in range(scheme['match_es'], scheme['amb5_3']+1):
				if d[i] != "": p[ischeme[i]] = d[i]

		# rating data
		if wave not in rating: rating[wave] = {}
		(mid, wid) = (pid, partner) if gender == 'm' else (partner, pid)
		if (mid, wid) not in rating[wave]:
			rating[wave][mid, wid] = {}
		date = rating[wave][mid, wid]
		for i in range(scheme['dec'], scheme['met']+1):
			if d[i] != "": date[gender + ischeme[i]] = d[i]

	return people, rating


random_vector = lambda k: [random.uniform(-1, 1) * 0.1 for _ in range(k)]
MSE = lambda true, pred: sum((a-b)**2 * 1. /len(true)  for a,b in zip(true,pred))
inner = lambda X,Y: sum(x*y for x,y in zip(X,Y))

def do_training(rating, attr='sinc', lamBw=0, lamBm=0, \
				lamGw=10**0.22, lamGm=10**0.22, Niter=1, niter=7, k=3):
	# preprocessing
	R = {}
	M = collections.defaultdict(set)
	W = collections.defaultdict(set)
	for wave in rating:
		for m, w in rating[wave]:
			d = rating[wave][m,w]
			if 'm' + attr in d:
				M[(wave,w)].add((wave,m))
				W[(wave,m)].add((wave,w))
				R[(wave,m), (wave,w)] = float(d['m' + attr])

	X = R.keys()
	y = [R[m,w] for m,w in X]

	alpha = sum(y) * 1. / len(y)
	betaM = collections.defaultdict(float)
	betaW = collections.defaultdict(float)
	gammaM = collections.defaultdict(lambda: [0] * k)
	gammaW = collections.defaultdict(lambda: [0] * k)

	for w in M: gammaW[w] = random_vector(k)
	for m in W: gammaM[m] = random_vector(k)

	# alternating least square
	MSEs = [MSE(y, [alpha] * len(y))]
	for p in range(2 * Niter):
		for _ in range(niter):
			# update alpha
			alpha = sum(R[m,w] - betaM[m] - betaW[w] - inner(gammaM[m], gammaW[w]) for m, w in R) * 1. / len(R)
			# update beta_m
			for m in W:
				betaM[m] = sum(R[m,w] - alpha - betaW[w] - inner(gammaM[m], gammaW[w]) for w in W[m]) * 1. / (lamBm + len(W[m]))
			# update beta_i
			for w in M:
				betaW[w] = sum(R[m,w] - alpha - betaM[m] - inner(gammaM[m], gammaW[w]) for m in M[w]) * 1. / (lamBw + len(M[w]))
			
			if p % 2:
			# update gamma_i
				for w in M:
					for j in range(k):
						gammaW[w][j] = sum((R[m,w] - alpha - betaW[w] - betaM[m]) * gammaM[m][j] for m in M[w]) * 1.\
							 / (lamGw + sum(gammaM[m][j] for m in M[w]))
			else:
			# update gamma_u
				for m in W:
					for j in range(k):
						gammaM[m][j] = sum((R[m,w] - alpha - betaW[w] - betaM[m]) * gammaW[w][j] for w in W[m]) * 1.\
							 / (lamGm + sum(gammaW[w][j] for w in W[m]))

		# compute MSE
		MSEs += MSE(y, [alpha + betaW[w] + betaM[m] + inner(gammaW[w],gammaM[m]) for m, w in X]),
	pylab.plot(range(len(MSEs)), MSEs, '-b')
	pylab.show()
	print MSEs[-1]
	print MSE(y, [alpha] * len(y))
	return alpha, betaM, betaW, gammaM, gammaW

people, rating = readData()
alpha, betaM, betaW, gammaM, gammaW = do_training(rating)