# -*- coding: utf-8 -*-
import scrapy
from ..items import MyLyricsItem


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


class LyricsSpider2(scrapy.Spider):
    name = 'lyrics2'
   # allowed_domains = ['lyrics.com']
  #  start_urls = ['http://lyrics.com/']
    page_number = 0
    href_final = []

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
            LyricsSpider2.page_number += 1
            yield response.follow(str(next_button[0]), callback=self.parse_second_web, meta={"href": href_2, "next_button": next_button})
        else:
            for lists in LyricsSpider2.href_final:
                for ref in lists:
                    yield response.follow(str(ref), callback=self.parse_lyrics_2)

    def parse_lyrics_2(self, response):
        lyrics = response.css(".verse").css("::text").extract()
        artist_name = response.css(".banner-heading a").css("::text").extract()
        song_name = response.css("h1").css("::text").extract()

        yield{"lyrics": lyrics,
              "song_name": song_name[0],
              "artist": artist_name[0]
              }
