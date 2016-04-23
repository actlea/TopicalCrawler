
def clean_file(name):
	with open(name) as fr:
		content = fr.read()
		lines = content.splitlines()
	with open(name, 'w') as fw:
		for line in lines:
			tmp = line.split('\t')
			if len(tmp)<2:
				continue
			if len(tmp[1])<2:
				continue
			fw.write('%s\n' %line)

# clean_file('train-8-zh.txt')

# from sklearn.linear_model import SGDClassifier
# X = [[0., 0.], [1., 1.]]
# y = [0, 1]
# clf = SGDClassifier(loss="hinge", penalty="l2")
# clf.fit(X, y)
# clf.predict([[2., 2.]])


def work1(name):
	url2id={}
	with open(name) as fr:
		content = fr.read()
		lines = content.splitlines()
	
		for line in lines:
			tmp = line.split('\t')
			if len(tmp)<2:
				continue
			if len(tmp[1])<2:
				continue
			if tmp[1] in url2id:
				url2id[tmp[1]] += 1
			else:
				url2id[tmp[1]] = 1
	for url, num in url2id.items():
		if num>1:
			print url

work1('test-9-zh.txt')