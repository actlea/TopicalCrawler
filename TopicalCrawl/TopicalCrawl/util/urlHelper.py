#!/usr/bin/env python
#-*- coding: utf-8 -*-


import re


def stripNonAlphaNum(text):
    import re
    return re.compile(r'\W+', re.UNICODE).split(text)

def stripTags(pageContents):
    startLoc = pageContents.find("<hr/><h2>")
 
    pageContents = pageContents[startLoc:]

    inside = 0
    text = ''
    for char in pageContents:
        if char == '<':
            inside = 1
        elif (inside == 1 and char == '>'):
            inside = 0
        elif inside == 1:
            continue
        else:
            text += char
    return text


def preprocess_url(url):
	if not isinstance(url, unicode):
		url = url.decode('utf-8')
	return stripNonAlphaNum(url)


url = 'http://programminghistorian.org/lessons/output-keywords-in-context-in-html-file'
print preprocess_url(url)










