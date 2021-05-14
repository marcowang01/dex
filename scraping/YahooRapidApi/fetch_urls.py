import pandas as pd
import config
import json
import requests
import csv
from tqdm import tqdm

def fetch_urls(ticker):
    # query first page of results from yahoo api
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/news/v2/list"

    querystring = {"region": "US", "snippetCount": "28", "s": ticker}

    payload = ""
    headers = {
        'content-type': "text/plain",
        'x-rapidapi-key': config.RapidAPI_KEY,
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }

    raw = requests.request("POST", url,
                            data=payload,
                            headers=headers,
                            params=querystring)
    response = json.loads(raw.text)
    # with open('urls.json', 'w') as json_file:
    #     json.dump(response, json_file)

    # grabs uuids of the first 200 results
    uuids = []
    for article in response['data']['main']['stream']:
        uuids.append(article['id'])
    # parses string into json object (dict) to extract uuids
    chars_to_skip = len("paginationString=")
    pagination_string = response['data']['main']['pagination']['uuids']
    pagination_uuids = json.loads(pagination_string[chars_to_skip:])

    for uuid in pagination_uuids['contentOverrides'].keys():
        uuids.append(uuid)

    # fetch url and date from yahoo api
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/news/v2/get-details"
    headers = {
        'x-rapidapi-key': config.RapidAPI_KEY,
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }

    with open('urls.csv', 'w', newline='') as csvfile:
        fieldnames = ['date', 'source', 'title', 'url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for uuid in tqdm(uuids):
            querystring = {"uuid": uuid, "region": "US"}
            raw = requests.request("GET", url,
                                        headers=headers,
                                        params=querystring)
            # print(uuid)
            response = json.loads(raw.text)
            contents = response['data']['contents'][0]['content']
            writer.writerow({
                'date': contents['pubDate'],
                'source': contents['provider']['displayName'],
                'title': contents['title'],
                'url': contents['canonicalUrl']['url']
            })


if __name__ == '__main__':
    fetch_urls(config.search_term)
