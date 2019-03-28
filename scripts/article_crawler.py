import sys

from scrapy.crawler import CrawlerProcess

from articlespider import *

class ArticleCrawler:
    """

    """

    def start(self, urls):
        """
        Starts the ArticleSpider with the given urls.
        :return: None
        """
        process = CrawlerProcess()
        process.crawl(ArticleSpider, start_urls=urls)
        process.start()


if __name__ == '__main__':
    path = sys.argv[1]
    urls = pd.read_csv(path, header=None, squeeze=True).tolist()
    ac = ArticleCrawler()
    ac.start(urls)