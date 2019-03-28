import dateutil

import pandas as pd
import scrapy


class ArticleSpider(scrapy.Spider):
    """

    """

    name = "articlespider"

    def __init__(self, start_urls=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls

    def get_content(self, response):
        """
        Retrieves the article text and metadata. Metadata consists of date of issue, headline and url.

        :param response: scrapy.http.TextResponse The response object.
        :return:         pandas.DataFrame         The article content.
        """
        content = {}
        content["date"] = int(dateutil.parser.parse(response.xpath("(//time)[1]/@datetime").extract()[0]).timestamp())
        content["headline"] = response.xpath("//div[@class='headline-container-inner']/h1/text()").extract()
        content["text"] = (' '.join(
            response.xpath("//article//text()").extract())
        ).replace('\n', '')
        content["url"] = response.request.url
        return content

    def start_requests(self):
        """
        Starts the requests with the urls of the spider
        """
        for url in self.start_urls:
            yield scrapy.Request(callback=self.parse, url=url)

    def parse(self, response):
        """
        Retrieves the content of the articles.

        :param response: scrapy.http.TextResponse The response object.
        """
        content = self.get_content(response)

        with open('dataset3_articles.tsv', 'a') as f:
            pd.DataFrame(content).to_csv(f, header=f.tell() == 0, sep='\t')
