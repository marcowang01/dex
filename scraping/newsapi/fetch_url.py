from newsapi import NewsApiClient
import pandas as pd
import config
import json


def fetch_urls():
    newsapi = NewsApiClient(api_key=config.NewsAPI_KEY)
    sources = newsapi.get_sources(category='business')
    sources_str = ""
    for s in sources['sources']:
        sources_str += s['id']
        sources_str += ','

    for t in config.tickers:
        all_articles = newsapi.get_everything(q=t,
                                              sources='bloomberg,the-wall-street-journal,reuters,',
                                              from_param='2021-04-02',
                                              to='2021-05-01',
                                              language='en',
                                              sort_by='relevancy',
                                              page=1)

        with open('person.json', 'w') as json_file:
            json.dump(all_articles, json_file)


if __name__ == '__main__':
    fetch_urls()
