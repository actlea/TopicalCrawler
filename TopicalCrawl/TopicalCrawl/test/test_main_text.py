#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: test_main_text.py
@time: 16-3-4 上午8:49
"""
#使用编辑距离来测试抽取文档的准确性

import sys
import re
import os
import copy

from TopicalCrawl.util import StringHelper, FileHelper
from api import HtmlHelper
from cetd import get_aricle_cetd


def chinese_segment(content):
    return StringHelper.chinese_segment(content)


#最长公共子串
def LCS(a=[[]], b=[[]]):
    try:
        lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x[0] == y[0]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
                    # read the substring out from the matrix
            # result = ''
        result = []
        x, y = len(a), len(b)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x-1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y-1]:
                y -= 1
            else:
                assert a[x-1][0] == b[y-1][0]
                # result = a[x-1] + result
                result.append(a[x-1][0])
                x -= 1
                y -= 1
        return result[::-1]
    except MemoryError:
        return None

#输入文件，返回经过处理之后的文件内容
def preprocess(orig_sequence, contestant_raw, benchmark_raw):

    def preprocessing_action(versioncode, whosetext, input_lines):
        output_lines=[]
        for line in input_lines:
            line = chinese_segment(line)
            if versioncode=='_':
                # output_lines.append(line)
                pass
            elif versioncode=='b': #Removing BLANK lines
                line = line.replace('<h>','')
                line = line.replace('<p>','')
                line = line.replace('<li>','')
                # if line.strip()!='':
                #     output_lines.append(line)
            elif versioncode=='r': #REMOVE markup tags
                pass
                # output_lines.append(line) #暂时未实现
            elif versioncode=='t':#TOKENISE file
                line = line.replace('\t',' ') #Replace tabs by spaces
                line = line.replace('  ',' ') ##Remove multiple spaces
                line = ''.join([l+'\n' for l in line.splitlines()]) #replace oddball end-of-line markers with \n

                line = line.replace(' \n','\n') #Remove spaces before end-of-lines
                line = line.lstrip() #Remove spaces at beginning of lines
            elif versioncode == 'n': #NORMALISE file,去除标点符号
                line = ''.join(re.split(',|;|:|\.|\?|!', line)) #Remove all punctuation. THIS LIST CAN BE INCREASED.
                line = line.lower()
                # output_lines.append(line)
            elif versioncode == 'v':#VERTICALISE file
                line = line.replace(' ', '\n')#Replace all spaces with a new line
                # output_lines.append(line)
            else:
                sys.exit("You have input an incorrect code for pre-processing. ABORTING!!\n")

            if line.strip()!='':
                output_lines.append(line)
        return output_lines

    preprocess_sequence = orig_sequence
    input1 = open(contestant_raw, 'r')
    input2 = open(benchmark_raw, 'r')

    contest_content_lines = input1.readlines()
    bench_content_lines = input2.readlines()

    if len(contest_content_lines)==0 or len(bench_content_lines)==0:
        return (None, None, None, None)

    contestant_pre = preprocessing_action("_","C",contest_content_lines)
    benchmark_pre = preprocessing_action("_","B",bench_content_lines)
    contestant_post = contestant_pre
    benchmark_post = benchmark_pre

    for s in preprocess_sequence:
        if s=='n':
            contestant_post = preprocessing_action(s, "C", contestant_post)
            benchmark_post = preprocessing_action(s, "B", benchmark_post)
        else:
            contestant_pre = preprocessing_action(s, "C", contestant_pre)
            contestant_post = preprocessing_action(s, "C", contestant_post)
            benchmark_pre = preprocessing_action(s, "B", benchmark_pre)
            benchmark_post = preprocessing_action(s, "B", benchmark_post)

    input2.close()
    input1.close()
    return (''.join(contestant_pre), ''.join(contestant_post), ''.join(benchmark_pre), ''.join(benchmark_post))

#根据文件内容，创建向量
def buildtextarray(parameters):
    (whosetext,localnormalised,localprenorm) = parameters
    localarray = []

    for line in localnormalised.splitlines():
        line = line.replace('\n','')
        localarray.append(line)

    templocalarray = len(localarray)
    arraycounttemp = 0

    localprenorm_len = len(localprenorm.splitlines())
    for line in localprenorm.splitlines():
        if arraycounttemp>=templocalarray:
            print
            break
            # sys.exit('Original and Procesed array not equal length\n')
        line = line.replace('\n','')
        localarray[arraycounttemp] = [localarray[arraycounttemp],line]
        arraycounttemp += 1

    return localarray


class Score:
    def __init__(self, benchmark_file, contestant_file, orig_sequence = 'btnv'):
        self.benchmark_file  = benchmark_file
        self.contestant_file = contestant_file
        self.orig_sequence = orig_sequence

    def longest_common_sequence_score(self):
        filename = self.benchmark_file.split('/')[-1]
        (contestant_pre, contestant_post, benchmark_pre, benchmark_post) = preprocess(self.orig_sequence,
                                                                                      self.contestant_file,
                                                                                      self.benchmark_file
        )
        if contestant_pre is None or benchmark_pre is None:
            return

        contestantparametres = ("C", contestant_post, contestant_pre)
        contestant = buildtextarray(contestantparametres) #Build array of contestants' text
        #2.BUILD ARRAY OF BENCHMARK TEXT
        benchmarkparametres = ("B",benchmark_post,benchmark_pre)
        benchmark = buildtextarray(benchmarkparametres)

        result = LCS(contestant, benchmark)
        if result is None:
            return

        #4.precision
        P = 1.0*len(result)/len(contestant)
        R = 1.0*len(result)/len(benchmark)
        PR= P+R
        if PR==0:PR=1
        F1 = 2.0*P*R/PR

        score = 1.0*len(result)/(len(contestant)+len(benchmark)-len(result))

        print "Cont:%s, Benc:%s" %(self.contestant_file.split('/')[-1], self.benchmark_file.split('/')[-1]),
        print "Precision: %.4f; Recall: %.4f; F1: %.4f; Score: %.4f" %(P, R, F1, score)

        return P, R, F1, score

def get_score(benchmark_file, contestant_file):
    s = Score(benchmark_file, contestant_file)
    return s.longest_common_sequence_score()


def csvout(src_file):
    pres, recall,fscore, escore, d = [], [], [], [], {}
    with open(src_file) as fr:
        lines = fr.readlines()
        lines = lines[1:]
        for line in lines:
            line = line.strip()
            line = line.split()
            pres.append(line[2])
            recall.append(line[3])
            fscore.append(line[4])
            escore.append(line[5])
            d[line[0]] = (float(line[2]),float(line[3]), float(line[4]), float(line[5]))

    with open(src_file.split('.')[0] + '_P.csv', 'w') as P:
        P.write(','.join(pres))
    with open(src_file.split('.')[0] + '_R.csv', 'w') as R:
        R.write(','.join(recall))
    with open(src_file.split('.')[0] + '_F.csv', 'w') as F:
        F.write(','.join(fscore))
    with open(src_file.split('.')[0] + '_S.csv', 'w') as S:
        S.write(','.join(escore))

    float_epsilon = 0.01

    ipres = [float(i) for i in pres if float(i) > float_epsilon]
    irres = [float(i) for i in recall if float(i) > float_epsilon]
    ifres = [float(i) for i in fscore if float(i) > float_epsilon]
    iscore = [float(i) for i in escore if float(i) > float_epsilon]

    ipres.sort()
    irres.sort()
    ifres.sort()
    iscore.sort()
    p_avg = sum(ipres[1:-1]) / (len(ipres) - 2)
    r_avg = sum(irres[1:-1]) / (len(irres) - 2)
    f_avg = sum(ifres[1:-1]) / (len(ifres) - 2)
    s_avg = sum(iscore[1:-1]) / (len(iscore) - 2)

    with open(src_file.split('.')[0] + '_AVG.txt', 'w') as f:
        f.write('pavg:%.2f\travg:%.2f\tfavg:%.2f\tsavg:%.2f\n' %(p_avg, r_avg, f_avg, s_avg));



def compute(t_dir): #cetd/
    contestant_dir = t_dir+'cerd/'
    origin_dir = t_dir + 'original/'
    contestant_dir1 = t_dir + 'cetd/'

    ext=lambda x: int(x.split('.')[0])

    origin_list = os.listdir(origin_dir)#网页原码
    origin_list.sort(cmp=lambda x,y:ext(x)-ext(y))
    if not os.path.exists(contestant_dir):
        os.mkdir(contestant_dir)
    if not os.path.exists(contestant_dir1):
        os.mkdir(contestant_dir1)


    # flag = False
    for _file in origin_list:
        # if file=='72.html': flag = True
        # if not flag: continue
        print '%s is parsing....' %_file
        filename = origin_dir+_file
        content = FileHelper.readFile(filename)


        doctree = HtmlHelper.create_doc(content, 'utf-8')
        cetd_doc = copy.deepcopy(doctree)

        # cetr算法抽取网页正文
        doctree = HtmlHelper.pre_process_domtree(doctree)
        article,title_text = HtmlHelper.get_article(doctree)
        article = title_text+' '+article
        outFile = contestant_dir + 'C_'+_file.split('.')[0]+'.txt'
        FileHelper.WriteInUTF8(outFile, article)

        # cetd算法抽取网页正文
        article_c = get_aricle_cetd(cetd_doc)
        outFile = contestant_dir1 + 'C_' + _file.split('.')[0] + '.txt'
        FileHelper.WriteInUTF8(outFile, article_c)


def mapping(cetr_list, gold_file_list):
    ig, ir = 0, 0
    mapping_files = []
    for ir in range(len(cetr_list)):
        if 'C_' + gold_file_list[ig] == cetr_list[ir] and ig < len(gold_file_list):
            mapping_files.append((gold_file_list[ig], cetr_list[ir]))
            ig += 1
    return mapping_files


def check_result(t_dir):
    gold_dir = t_dir + 'gold/'
    cerd = t_dir+'cerd/'
    cetd = t_dir + 'cetd/'
    result = t_dir + 'result/'
    if not os.path.exists(result):
        os.mkdir(result)

    gold_file_list = os.listdir(gold_dir)  # 手动标记的数据集
    cerd_list = os.listdir(cerd)
    cetd_list = os.listdir(cetd)
    # 将手动标注和算法抽取的文件名一一对应

    mapping_cerd = mapping(cerd_list, gold_file_list)
    mapping_cetd = mapping(cetd_list, gold_file_list)



    ##计算算法抽取的文本与手动标注的文本的lsc score
    print '=============================='
    print 'cerd'
    with open(result+'cerd.txt', 'w') as f1:
        headers = '{:<10} {:<15} {:^15} {:^15} {:^15} {:^15}\n'.format(
            'Gold', 'Origin', 'Precision', 'Recall', 'F1', 'Score')
        f1.write(headers)
        for files in mapping_cerd:
            print files[1]
            gold_file= gold_dir + files[0]
            cont_file = cerd + files[1]
            P, R, F1, score = get_score(gold_file,cont_file )
            data = '{:<10} {:<15} {:^15.4} {:^15.4} {:^15.4} {:^15.4}\n'.format(files[0], files[1],P, R, F1, score)
            f1.write(data)
    print '=============================='
    print 'cetd'
    with open(result + 'cetd.txt', 'w') as f2:
        headers = '{:<10} {:<15} {:^15} {:^15} {:^15} {:^15}\n'.format(
            'Gold', 'Origin', 'Precision', 'Recall', 'F1', 'Score')
        f2.write(headers)
        for files in mapping_cetd:
            print files[1]
            gold_file = gold_dir + files[0]
            cont_file = cetd + files[1]
            P, R, F1, score = get_score(gold_file, cont_file)
            data = '{:<10} {:<15} {:^15.4} {:^15.4} {:^15.4} {:^15.4}\n'.format(gold_file, cont_file, P, R, F1, score)
            f2.write(data)

    csvout(result+'cetr.txt')
    csvout(result + 'cetd.txt')



if __name__ == '__main__':
    home = '/mnt/UbutunShare/Work/CETD_DATA/'
    files = ['arstechnica/','BBC/', 'Chaos/', 'nytimes/', 'wiki/', 'YAHOO!/', 'Test/']

    compute(home+files[6])
    check_result(home+files[6])

