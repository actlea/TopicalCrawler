#!/usr/bin/env python
#-*- coding: utf-8 -*-


import os
import codecs
import re
import chardet
import codecs
import copy
import random

def get_unicode(content):
	ret = chardet.detect(content)
	encoding, confidence = ret['encoding'], ret['confidence']
	# print encoding, confidence	
	# return encoding
	content = content.decode(encoding, errors = 'ignore')
	



def clean(content):
	content = re.sub(r'\t','', content)				
	content = re.sub(r'[\r,\n]\s','',content)
	content = content.replace('&nbsp', '')
	content = ''.join([x.strip() for x in content if len(x.strip())!=0])
	return content


#从文件中读取文本，每一个文件写入一行，作为一个测试数据
def create_train_from_files(path,output):	
		with open(output, 'w') as fw:
			for f in os.listdir(path):
				print f
				f = os.path.join(path, f)
				inp = open(f, 'r')
				content = inp.read()
				tmp = copy.deepcopy(content)
				try:
					content = content.decode('GB2312', errors='ignore')
				except UnicodeDecodeError:
					print '%s error' %f
					continue

					# content = tmp.decode('GB2312')
				# except UnicodeDecodeError:
				# 	content = tmp.decode('gbk')
				# except:
				# 	# content = get_unicode(tmp)
				# 	continue

				content  = clean(content).encode('utf-8','ignore')
				inp.close()
				
				fw.write('%s\n' %content)

#将多个测试数据写入一个文档中，每行标记为label:text作为标记之后的测试数据
def label_file(in_file, train_file, test_file, label, mode='a', start=0, length=2000):
	
	fr = open(in_file, 'r')
	lines = fr.readlines()
	random.shuffle(lines)
	
	with open(train_file, mode) as fw:
		for line in lines[start:start+length]:
			line = line.strip()
			fw.write('%s\t%s\n' %(label, line))
	with open(test_file, mode) as fw2:
		for line in lines[start+length: start+2*length]:
			line = line.strip()
			fw2.write('%s\t%s\n' %(label, line))
	fr.close()


classFile = 'E:/ZJM/graduate/DataSet/Big/'
sport_file = classFile + 'C000014.txt'	
health_file = classFile+'C000013.txt'
finace_file = classFile+'C000008.txt'
it = classFile + 'C000010.txt'
yule = classFile + 'yule.txt'
car = classFile + 'C000007.txt'
lvyou = classFile + 'C000016.txt'
jiaoyu = classFile + 'C000020.txt'
zhaopin = classFile + 'C000022.txt'
wenhua = classFile + 'C000023.txt'
junshi = classFile + 'C000024.txt'


file_dic = {
	'0':sport_file, '1':health_file, '2':finace_file,
	'3':it, '4':yule, '5':car, '6':lvyou, '7':jiaoyu,
	'8':zhaopin, '9':wenhua, '10':junshi
}

test_file = 'E:/ZJM/TopicalCrawl/TopicalCrawl/classifier/sample-data/test-1-zh.txt'
train_file = 'E:/ZJM/TopicalCrawl/TopicalCrawl/classifier/sample-data/train-1-zh.txt'


def work1():
	label_file(sport_file, train_file, test_file, 0, mode='a', start=0, length = 2000)

	for label, name in file_dic.items():	
		print name
		if label !='0':
			label_file(name, train_file, test_file, 1, mode='a', start=0, length = 200)


# work1()
#move file
train_file = 'E:/ZJM/TopicalCrawl/TopicalCrawl/classifier/sample-data/train-2-zh.txt'
test_file = 'E:/ZJM/TopicalCrawl/TopicalCrawl/classifier/sample-data/test-2-zh.txt'

in_file = 'E:/ZJM/TopicalCrawl/TopicalCrawl/classifier/sample-data/anchor.txt'

def isdigit(s):
	for i in s:
		if i not in '0123456789\t':
			return True
	return False



def work3(in_file, out_file):
	with open(in_file) as fr:
		content = fr.read()
		lines = content.splitlines()
	with open(out_file, 'w') as fw:
		for line in lines:
			tmp=line.split('\t')[1]			
			u=tmp.decode('utf-8')
			if len(u)>10 and isdigit(u):
				fw.write('%s\n' %line)

home = 'E:/ZJM/TopicalCrawl/TopicalCrawl/classifier/sample-data/'
# work3(home+'webpage.sample.txt', home+'anchor.txt')
# work3(home+'train-anchor-zh.txt', home+'train-anchor-zh.txt')


from collections import Counter
def work2(in_file, train_file, test_file, length=7000):
	P, N=[], []

	with open(in_file) as fr:
		content = fr.read()
		for line in content.splitlines():
			tmp = line.strip()
			tmp = line.split('\t')
			label = tmp[0]
			if label=='0':
				P.append(tmp[1])
			else:
				N.append(tmp[1])
	
	random.shuffle(P)
	random.shuffle(N)

	A=P[:length]
	B=N[:length]
	C=P[length:2*length]
	D=N[length:2*length]

	with open(train_file, 'w') as f1:
		for a in A:
			f1.write('%s\t%s\n' %('0',a))
		for b in B:
			f1.write('%s\t%s\n' %('1',b))

	with open(test_file, 'w') as f2:
		for c in C:
			f2.write('%s\t%s\n' %('0',c))
		for d in D:
			f2.write('%s\t%s\n' %('1',d))
		
# work2(in_file, train_file, test_file, length=5000)
def readfile(_file):
	with open(_file) as fr:
		content = fr.read()
		return content.splitlines()

def work4(pos, neg, train_file, mode='a'):
	P,N=[],[]
	with open(pos) as fr:
		content = fr.read()
		for line in content.splitlines():
			tmp = line.split('\t')
			try:
				P.append(tmp[1])
			except:
				continue

	with open(neg) as f1:
		content = f1.read()
		for line in content.splitlines():
			tmp = line.split('\t')
			try:
				N.append(tmp[1])
			except:
				continue

	with open(train_file, mode) as f2:
		for p in P:
			f2.write('0\t%s\n' %p)
		for n in N:
			f2.write('1\t%s\n' %n)
home = 'E:/ZJM/TopicalCrawl/TopicalCrawl/classifier/sample-data/source/'

work4(home+'pos-url-anchor.txt',home+'neg-url-anchor.txt', home+'train-anchor.txt')

def work5(infile, outfile, mode='a'):
	lines = readfile(infile)
	with open(outfile, mode) as f:
		for line in lines:
			tmp=line.split('\t')
			try:
				label='1'
				url = tmp[1]
				f.write('%s\t%s\n' %(label, url))
			except IndexError:
				print tmp
				continue

# work5(home+'source/url.neg.txt', home+'url.txt')
home = 'E:/ZJM/graduate/DataSet/dic/'






