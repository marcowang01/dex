import scrapy
from yahoonews.yahoonews.items import YahoonewsItem
from scrapy.http import request
import config

class ArticlesSpider(scrapy.Spider):

    #name of spider
    name = "articles"

    # domains in which the spider can operate
    allowed_domains = ["finance.yahoo.com"]
    # list of urls to be scraped
    urls = [""]
    start_urls = urls
    
    def start_requests(self):

        for t in config.tickers:

            # Hardcoded URL that returns stock symbol related articles
            url = f"https://finance.yahoo.com/m/{t}?.html"

        link_urls = [url.format(i) for i in range(0,500)]

        # Loops through 500 pages to get the article links
        for link_url in link_urls: 
            print(link_url)

        


    def parse(self, response):
        # the parse method is called by default on each url of the
        # start_urls list 
        item = YahoonewsItem()
        # the date, keywords and body attributes are retrieved from
        # the response page using the XPath query language
        item['date'] = response.xpath('//meta[@content][@name="pub_date"]/@content').extract()
        item['keywords'] = response.xpath('//meta[@content][@name="keywords"]/@content').extract() 
        item['body'] = response.xpath('//div[@id = "article_body"]/p/text()').extract()
        # the complete item filled with all its attributes 
        yield item

