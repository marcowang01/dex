import requests
import config


class Get_news():
    def __init__(self, symbol):
        self.symbol = symbol

        
    def lookup_url():   
        url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/news/v2/list"

        for t in config.tickers: 
            querystring = {"region":"US","snippetCount":"28","s":f"{t}"}

        payload = "Pass in the value of uuids field returned right in this endpoint to load the next page, or leave empty to load first page"
        headers = {
            'content-type': "text/plain",
            'x-rapidapi-key': f"{config.RapidAPI_KEY}",
            'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
            }

        response = requests.request("POST", url, data=payload, headers=headers, params=querystring)
        print(response.text)

Get_news.lookup_url()