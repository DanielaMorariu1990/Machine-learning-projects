# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class GetLyricsItem(scrapy.Item):
    # define the fields for your item here like:
    artist = scrapy.Field()
    song_name = scrapy.Field()
    lyrics = scrapy.Field()
    year = scrapy.Field()
    album_name = scrapy.Field()
    views = scrapy.Field()
    likes = scrapy.Field()
    dislikes = scrapy.Field()
    # pass
