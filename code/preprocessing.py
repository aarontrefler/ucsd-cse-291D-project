import csv

mean = lambda x : sum(x) * 1. / len(x)
var  = lambda x : mean([t**2 for t in x]) - mean(x) ** 2
MSE = lambda true, pred: sum((a-b)**2 * 1. /len(true)  for a,b in zip(true,pred))
accuracy = lambda x, x_: sum(t == t_ for t, t_ in zip(x, x_)) * 1. / len(x)
attrs = ['attr', 'sinc', 'fun', 'amb', 'intel', 'shar']

def readData(path):

	def isNumeric(s):
		s = s.split('.')
		return len(s) <= 2 and all(d.isdigit() for d in s)

	with open(path, 'rU') as csvfile:
		data = [r for r in csv.reader(csvfile)]
		scheme = {t: i for i, t in enumerate(data[0])}
		ischeme = {i: t for t, i in scheme.items()}
		data = [map(lambda s: float(s) if isNumeric(s) else s, d) for d in data[1:]]

	return data, scheme, ischeme


def formatData(data, scheme, ischeme):

	def newPerson():
		return {f+'s':[] for f in attrs[:-1]}

	M, W, R = {}, {}, {}
	for d in data:
		wave = d[scheme['wave']]
		gender = 'm' if d[scheme['gender']] != 0 else 'w'
		rater = (wave, d[scheme['id']])
		ratee = (wave, d[scheme['partner']])

		# personal data
		P = M if gender == 'm' else W
		if rater not in P:
			P[rater] = newPerson()
			for i in range(scheme['age'], scheme['amb5_1']+1) + range(scheme['match_es'], scheme['amb5_3']+1):
				if d[i] != "": P[rater][ischeme[i]] = d[i]

		# rating data
		pair = (rater, ratee) if gender == 'm' else (ratee, rater)
		if pair not in R: R[pair] = {}
		R[pair]['samerace'] = d[scheme['samerace']]
		for i in range(scheme['dec'], scheme['met']+1):
			if d[i] != "": R[pair][gender + ischeme[i]] = d[i]

	# more statistics
	R = {(m,w):R[m,w] for m,w in R if m in M and w in W}
	for m,w in R:
		for f in attrs[:-1]:
			if 'm'+f in R[m,w]:
				W[w][f+'s'] += R[m,w]['m'+f],
			if 'w'+f in R[m,w]:
				M[m][f+'s'] += R[m,w]['w'+f],

	return M, W, R

data, scheme, ischeme = readData('../data/raw/speed_dating_data.csv')
Men, Women, Ratings = formatData(data, scheme, ischeme)