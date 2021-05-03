from newsapi import NewsApiClient
import pandas as pd
import config
import json

# TODO: change to using yahoo API for fetching daily news

def fetch_urls():
    newsapi = NewsApiClient(api_key=config.NewsAPI_KEY)
    sources = newsapi.get_sources(category='business')
    sources_str = ""
    print("fetching sources...")
    for s in sources['sources']:
        sources_str += s['id']
        sources_str += ','

    print(f"fetching articles for {config.search_term}...")
    all_articles = newsapi.get_everything(q=config.search_term,
                                          from_param='2021-04-02',
                                          to='2021-05-02',
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=10,
                                          page=1)

    count = all_articles["totalResults"]
    article_list = []
    for i in range(count // 100 + 1):
        all_articles = newsapi.get_everything(q=config.search_term,
                                              from_param='2021-04-02',
                                              to='2021-05-02',
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=100,
                                              page=i + 1)

        article_list.append(all_articles['articles'])

    with open(f'{config.search_term}_urls.json', 'w') as json_file:
            json.dump(article_list, json_file)


if __name__ == '__main__':
    fetch_urls()
