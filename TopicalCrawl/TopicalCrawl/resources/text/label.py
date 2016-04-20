
with open('user_dict.txt') as fr:
	content = fr.read()
	lines = content.splitlines()
	fr.close()
with open('user_dict.txt', 'w') as fw:
	for i,line in enumerate(lines):
		if len(line.split(' '))>1:
			line = '.'.join(line.split(' '))
		fw.write('%s\n' %line)
