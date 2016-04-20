#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: config.py
@time: 16-3-2 下午7:22
"""

# =============================
# sports catagory





# ==============================
# data resources
kg_char_num="char-number"
kg_tag_num="tag-number"
kg_linkchar_num="linkchar-number"
kg_linktag_num="linktag-number"
kg_text_density="text-density"
kg_density_sum="density-sum"
kg_max_density_sum="max-density-sum"
kg_mark="mark"
kg_geometry="geometry"

# 元素的属性名称
number='kg_number'  # 节点的编号，先序遍历的顺序
label='kg_label'  # 节点的类别
kg_xpath='kg_xpath'  # 节点的路径
record_weight='kg_record_weight'  # 节点的区域权重
px='kg_x'  # 节点的层次
py='kg_y'  # 先序遍历节点的编号
kg_record_mark='record_mark'  # 节点是否是某个块的祖先节点
kg_text_mark='text_mark'  # 是否是正文节点

import sys

float_epsilon=sys.float_info.epsilon

TAG_WEIGHT={'a': 0.2, 'button': 0.2, 'select': 0.2, 'div': 0.9, 'blockquote': 0.5, 'p': 1, 'span': 0.5, 'li': 0.5}

TITLE_TAG=('h1', 'h2', 'h3', 'h4', 'strong')  # 修饰文章标题的标签,'strong', 'b', 'a','div',
MARKUP_TAG=('b', 'span', 'strong', 'em', 'i')

WEBSITE=[ 'sina', 'qq', 'souhu', 'ifeng', 'hupu' ]
LEIBIE=[ 'ouguan',  # 欧冠
         'yingchao',  # 英超
         'xijia',  # 西甲
         'yijia',  # 意甲
         'dejia',  # 德甲
         'fajia',  # 法甲
         'zhchao',  # 中超
         'yaguan',  # 亚冠
         'zhjia',  # 中甲
         'nvzu',  # 女足
         'nba',
         'cba',
         'zonghe',  # 综合
         ]

Other_URL=[
    'http://yule.sohu.com/',
    'http://ent.sina.com.cn/',
    'http://ent.news.cn/',
    'http://finance.people.com.cn/',
    'http://www.ftchinese.com/channel/economy.html',
    'http://www.ce.cn/',
    'http://mil.news.sina.com.cn/',
    'http://finance.sina.com.cn/',
    'http://tech.sina.com.cn/',
    'http://auto.sina.com.cn/',
    'http://blog.sina.com.cn/',
    'http://book.sina.com.cn/',
    'http://edu.sina.com.cn/',
    'http://fashion.sina.com.cn/',
    'http://city.sina.com.cn/',
    'http://travel.sina.com.cn/'
    'http://mil.sohu.com/',
    'http://history.sohu.com/',
    'http://yule.sohu.com/',
    'http://business.sohu.com/',
    'http://it.sohu.com/',
    'http://www.focus.cn/',
    'http://fashion.sohu.com/',
    'http://finance.qq.com/',
    'http://ent.qq.com/',
    'http://fashion.qq.com/',
    'http://auto.qq.com/',
    'http://house.qq.com/',
    'http://tech.qq.com/',
    'http://games.qq.com/'
]

Test_6_URL=[
    'http://sports.sina.com.cn/',
    'http://sports.sohu.com/',
    'http://sports.qq.com/',
    'http://sports.163.com/',
    'http://sports.people.com.cn/',
    'http://sports.baidu.com/',
    'http://www.hupu.com/',
    'http://sports.ifeng.com/',
    'http://www.news.cn/sports/',
    'http://www.ftchinese.com/channel/economy.html',
    'http://finance.qq.com/',
    'http://finance.sina.com.cn/',
    'http://money.163.com/',
    'http://business.sohu.com/',
    'http://finance.people.com.cn/',
    'http://finance.baidu.com/',
    'http://finance.ifeng.com/',
    'http://yule.sohu.com/',
    'http://ent.163.com/',
    'http://ent.qq.com/',
    'http://ent.sina.com.cn/',
    'http://ent.people.com.cn/',
    'http://yule.baidu.com/',
    'http://ent.ifeng.com/',
    'http://ent.news.cn/',
    'http://war.163.com/',
    'http://mil.sohu.com/',
    'http://military.people.com.cn/',
    'http://mil.news.baidu.com/',
    'http://news.ifeng.com/mil/',
    'http://www.news.cn/mil/',
    'http://mil.qq.com/',
    'http://auto.ifeng.com/',
    'http://www.news.cn/auto/',
    'http://auto.baidu.com/',
    'http://auto.qq.com/',
    'http://auto.people.com.cn/',
    'http://auto.163.com/',
    'http://auto.sina.com.cn/',
    'http://edu.qq.com/',
    'http://edu.people.com.cn/',
    'http://edu.163.com/',
    'http://edu.sina.com.cn/',
    'ttp://tech.ifeng.com/',
    'http://tech.qq.com/',
    'http://scitech.people.com.cn/',
    'http://tech.163.com/',
    'http://medicine.people.com.cn/',
    'http://health.sohu.com/'
]



Test6_Domains=[
    'sports.sina.com.cn',
    'sports.sohu.com',
    'sports.qq.com',
    'sports.163.com',
    'sports.people.com.cn',
    'sports.baidu.com',
    'www.hupu.com',
    'sports.ifeng.com',
    'www.news.cn',
    'www.ftchinese.com',
    'finance.qq.com',
    'finance.sina.com.cn',
    'money.163.com',
    'business.sohu.com',
    'finance.people.com.cn',
    'finance.baidu.com',
    'finance.ifeng.com',
    'yule.sohu.com',
    'ent.163.com',
    'ent.qq.com',
    'ent.sina.com.cn',
    'ent.people.com.cn',
    'yule.baidu.com',
    'ent.ifeng.com',
    'ent.news.cn',
    'war.163.com',
    'mil.sohu.com',
    'military.people.com.cn',
    'mil.news.baidu.com',
    'news.ifeng.com',
    'www.news.cn',
    'mil.qq.com',
    'auto.ifeng.com',
    'www.news.cn',
    'auto.baidu.com',
    'auto.qq.com',
    'auto.people.com.cn',
    'auto.163.com',
    'auto.sina.com.cn',
    'edu.qq.com',
    'edu.people.com.cn',
    'edu.163.com',
    'edu.sina.com.cn',
    'tech.ifeng.com',
    'tech.qq.com',
    'scitech.people.com.cn',
    'tech.163.com',
    'medicine.people.com.cn',
    'health.sohu.com',
    'http://health.sohu.com/'
]

Test7_URL=[
    'http://sports.sina.com.cn/',
    'http://sports.sohu.com/',
    'http://sports.qq.com/',
    'http://sports.163.com/',
    'http://sports.people.com.cn/',
    'http://sports.baidu.com/',
    'http://www.hupu.com/',
    'http://sports.ifeng.com/',
    'http://www.news.cn/sports/',
    'http://www.news.cn/sports/'
    'http://sports.cntv.cn/',
    'http://sports.tom.com/',
    'http://sports.eastday.com/',
    'http://www.chinanews.com/sports.shtml',
    'http://finance.qq.com/',
    'http://tech.163.com/',

]
Test7_Domain=[

]

SPORT=0
HEALTH=1
FINACE=2
