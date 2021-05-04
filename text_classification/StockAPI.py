from alpha_vantage.timeseries import TimeSeries
from config import StockAPI_KEY
from datetime import datetime, timedelta
import yfinance as yf
import sys, os


# Input API Key
key = StockAPI_KEY


# Looks up stock info based on given time
def get_delta(ticker, date1, date2):

    day_before, df_before = lookup_date_and_quote(ticker, date1, 0)
    day_after, df_after = lookup_date_and_quote(ticker, date2, 1)

    if df_before is None or df_after is None:
        return 0.00001

    stock_price_before = df_before["Close"][day_before]
    stock_price_after = df_after["Close"][day_after]

    # Compares stock prices before and after
    # return stock_price_after > stock_price_before:
    delta = float(stock_price_after) - float(stock_price_before)
    # print(round(delta, 3))
    return round(delta, 3)


# validates the date and returns a dataframe object of the quote of the day
def lookup_date_and_quote(ticker, date, direction):
    date1 = date


    count = 0
    while True:
        if count == 5:
            return "invalid date", None

        datetime_object = datetime.strptime(date1, '%Y-%m-%d')
        datetime_object += timedelta(days=1)
        date2 = datetime.strftime(datetime_object, '%Y-%m-%d')

        # prevents printing
        sys.stdout = open(os.devnull, 'w')
        df = yf.download(tickers=ticker, start=date1, end=date2, progress=False)
        sys.stdout = sys.__stdout__

        if len(df.index) == 1:
            return date1, df

        datetime_object = datetime.strptime(date1, '%Y-%m-%d')
        if direction > 0:
            datetime_object += timedelta(days=1)
        else:
            datetime_object -= timedelta(days=1)
        date1 = datetime.strftime(datetime_object, '%Y-%m-%d')
        count += 1


if __name__ == '__main__':
    # df = yf.download(tickers="SPY", start="2020-02-10", end="2020-02-11")
    # print(df["Close"])

    get_delta("SPY", "2020-02-07", "2020-02-08")












































    
    
    

