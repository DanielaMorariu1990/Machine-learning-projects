# -*- coding: utf-8 -*-
import scrapy
from ..items import GetLyricsItem


my_search = []
input_user = True
while input_user:
    search_item = input(
        "Which artist would you like to scrape? \n if you wish to stop type 'Done' ")
    if search_item.upper() != "DONE":
        my_search.append(search_item)
    if search_item.upper() == "DONE":
        input_user = False


class LyricsSpider(scrapy.Spider):
    name = 'lyrics'
    allowed_domains = ['lyrics.com']
  #  start_urls = ['http://lyrics.com/']

    def start_requests(self):

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
        item_song = GetLyricsItem()
        lyrics = response.css("#lyric-body-text").css("::text").extract()
        song_name = response.css("#lyric-title-text").css("::text").extract()
        artist_name = response.css(".lyric-artist > a").css("::text").extract()
        album_name = response.css(
            ".falbum .clearfix a").css("::text").extract()
        views = response.css(".info-views span").css("::text").extract()
        likes = response.css(".fa-thumbs-o-up+ span").css("::text").extract()
        dislikes = response.css(
            ".fa-thumbs-o-down+ span").css("::text").extract()

        item_song["lyrics"] = lyrics
        item_song["song_name"] = song_name[0]
        item_song["artist"] = artist_name[0]
        item_song["album_name"] = ("NA" if len(
            album_name) < 1 else album_name)[0]
        item_song["views"] = ("NA" if len(views) < 1 else views)[0]
        item_song["likes"] = ("NA" if len(likes) < 1 else likes)[0]
        item_song["dislikes"] = ("NA" if len(dislikes) < 1 else dislikes)[0]

        yield item_song
