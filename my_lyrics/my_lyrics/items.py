# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class MyLyricsItem(scrapy.Item):
    # define the fields for your item here like:
    song_name = scrapy.Field()
    artist = scrapy.Field()
    lyrics = scrapy.Field()
