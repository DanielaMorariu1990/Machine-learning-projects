import pandas as pd
from scrapy.crawler import CrawlerProcess
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from bs4 import BeautifulSoup
import json
from functools import reduce

# define input


def my_input():
    '''User inputs the artists that he wants to scrape
    from metrolyrics.com and lyrics.com
    '''
    my_search = []

    input_user = True
    while input_user:
        search_item = input(
            "Which artist would you like to scrape? \n if you wish to stop type 'Done' ")
        if search_item.upper() != "DONE":
            my_search.append(search_item)
        if search_item.upper() == "DONE":
            input_user = False

    return my_search


# write spiders


class LyricsSpider(CrawlSpider):
    '''This spider takes in the input of the user and scrapes
    the following information:


    #########results#####
    - artist name
    - artist album_name
    - artist song_name
    - artist lyrics

    '''
    name = 'lyrics'
    allowed_domains = ['lyrics.com']

  #  start_urls = ['http://lyrics.com/']

    def __init__(self, *a, **kw):
        super(LyricsSpider, self).__init__(*a, **kw)

    def start_requests(self):
        my_search = my_input()
        for search in my_search:
            search_new = "-".join(search.split(" "))
            yield scrapy.Request("https://www.lyrics.com/artist/" + search_new + "/", self.parse, meta={"search_art": search})

    def parse(self, response):

        href_lyrics = response.css(".qx a").css("::attr(href)").extract()

        for ref in href_lyrics:
            yield response.follow("https://www.lyrics.com" + str(ref), self.parse_lyrics)

    def parse_lyrics(self, response):
       # item = response.meta["item"]
       # yield item

        lyrics = response.css("#lyric-body-text").css("::text").extract()
        song_name = response.css("#lyric-title-text").css("::text").extract()
        artist_name = response.css(".lyric-artist > a").css("::text").extract()
        album_name = response.css(
            ".falbum .clearfix a").css("::text").extract()
        views = response.css(".info-views span").css("::text").extract()
        likes = response.css(".fa-thumbs-o-up+ span").css("::text").extract()
        dislikes = response.css(
            ".fa-thumbs-o-down+ span").css("::text").extract()

        yield{"lyrics": lyrics,
              "song_name": song_name[0],
              "artist": artist_name[0],
              "album_name": ("NA" if len(album_name) < 1 else album_name)[0],
              "views": ("NA" if len(views) < 1 else views)[0],
              "likes": ("NA" if len(likes) < 1 else likes)[0],
              "dislikes": ("NA" if len(dislikes) < 1 else dislikes)[0]}


class LyricsSpider2(CrawlSpider):
    '''This spider takes in the input of the user and scrapes
    the following information from metrolyrics.com:


    #########results#####
    - artist name
    - artist album_name
    - artist song_name
    - artist lyrics

    '''
    name = 'lyrics2'
   # allowed_domains = ['lyrics.com']
  #  start_urls = ['http://lyrics.com/']
    href_final = []

    def __init__(self, *a, **kw):
        super(LyricsSpider2, self).__init__(*a, **kw)

    def start_requests(self):

        my_search = my_input()
        for search in my_search:
            search_new = "-".join(search.split(" "))
            yield scrapy.Request("https://www.metrolyrics.com/" + search_new + "-lyrics.html", self.parse_second_web, meta={"search_art": search})

    def parse_second_web(self, response):

        href_2 = response.css("#popular .title").css("::attr(href)").extract()
        next_button = response.css(".next").css("::attr(href)").extract()
        LyricsSpider2.href_final.append(href_2)
        if next_button[0] != "javascript:void(0)":
            yield response.follow(str(next_button[0]), callback=self.parse_second_web, meta={"href": href_2, "next_button": next_button})
        else:
            for lists in LyricsSpider2.href_final:
                for ref in lists:
                    yield response.follow(str(ref), callback=self.parse_lyrics_2)

    def parse_lyrics_2(self, response):
        lyrics = reduce(lambda x, y: x+y,
                        response.css(".verse").css("::text").extract(), "1")
        artist_name = response.css(".banner-heading a").css("::text").extract()
        song_name = response.css("h1").css("::text").extract()

        yield{"lyrics": lyrics,
              "song_name": song_name[0],
              "artist": artist_name[0]
              }


process1 = CrawlerProcess({
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36',
    'FEED_FORMAT': 'json',
    'FEED_URI': 'output_LY.json',
})

process2 = CrawlerProcess({
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36',
    'FEED_FORMAT': 'json',
    'FEED_URI': 'output_metro2.json',
})


process1.crawl(LyricsSpider)
process2.crawl(LyricsSpider2)
process1.start()
process2.start()

######finishing the spidering process ####
