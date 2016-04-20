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

from TopicalCrawl.util import StringHelper, FileHelper
from api import HtmlHelper

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

        print 'result:','======'*15
        print "Cont:%s, Benc:%s" %(self.contestant_file.split('/')[-1], self.benchmark_file.split('/')[-1]),
        print "Precision: %.4f; Recall: %.4f; F1: %.4f; Score: %.4f" %(P, R, F1, score)

def get_score(benchmark_file, contestant_file):
    s = Score(benchmark_file, contestant_file)
    s.longest_common_sequence_score()

def test():
    contestant_dir = 'D:/OtherWork/CETD_DATA/BBC/TTTT/'
    gold_dir = 'D:/OtherWork/CETD_DATA/BBC/gold/'
    origin_dir = 'D:/OtherWork/CETD_DATA/BBC/original/'

    if not os.path.exists(contestant_dir):
        os.mkdir(contestant_dir)

    gold_file_list = os.listdir(gold_dir)
    origin_list = os.listdir(origin_dir)
    contestant_dir_list = os.listdir(contestant_dir)



    flag = False
    for _file in origin_list:
        if file=='72.html': flag = True
        if not flag: continue
        print '%s is parsing....' %_file
        filename = origin_dir+_file
        content = FileHelper.readFile(filename)

        doctree = HtmlHelper.create_doc(content, 'utf-8')
        doctree = HtmlHelper.pre_process_domtree(doctree)

        articles = HtmlHelper.get_article(doctree)

        outFile = contestant_dir + 'C_'+_file.split('.')[0]+'.txt'
        FileHelper.WriteInUTF8(outFile, articles)



    ig, ir = 0,0
    mapping_files = []
    for ir in range(len(contestant_dir_list)):
        if 'C_'+gold_file_list[ig]==contestant_dir_list[ir] and ig<len(gold_file_list):
            mapping_files.append((gold_file_list[ig], contestant_dir_list[ir]))
            ig += 1


    import time
    start = time.time()

    for files in mapping_files:
        print files[1]
        gold_file= gold_dir + files[0]
        ori_file = contestant_dir + files[1]
        get_score(gold_file,ori_file )
    print time.time()-start

if __name__ == '__main__':
    test()
