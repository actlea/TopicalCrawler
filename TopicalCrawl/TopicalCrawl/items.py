# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.item import Item, Field


class TopicalcrawlItem(scrapy.Item):
    pass

class UrlItem(Item):
    base_url = Field()
    url = Field()
    anchor_text = Field()
    neigb_text = Field()
    priority = Field()
    prob = Field()
    node_weight = Field()

class ResponseItem(Item):
    response = Field()
    count = Field()
    level0 = Field()
    level1 = Field()
    graph = Field()
    url_graph=Field()
    classifiers=Field()

class Document(Item):
    base_url = Field()
    title = Field()
    meta= Field()
    headlines = Field()
    para = Field()
    url_items = Field()
