from data.tiingo_data_fetcher import DataFetcher

fetcher = DataFetcher()

# pylint: disable=invalid-name
start_date = "2022-11-01"
end_date = "2022-12-31"
symbol = "AAPL"
token = "btcusd"

stock_data = fetcher.fetch_tiingo_stock_data(symbol, start_date, end_date)
print(stock_data.head())

token_data = fetcher.fetch_tiingo_crypto_data(token, start_date, end_date)
print(token_data.head())
