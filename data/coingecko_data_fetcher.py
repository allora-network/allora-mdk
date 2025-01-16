""""
This module contains a class for fetching cryptocurrency data from the CoinGecko API.
"""

import time

import pandas as pd
import requests


class CoingeckoDataFetcher:
    """
    A class to fetch cryptocurrency data from the CoinGecko API.
    """

    # Mapping from coin tickers to their respective CoinGecko API identifiers
    COIN_MAPPING = {
        "ETH": "ethereum",
        "BTC": "bitcoin",
        "USDT": "tether",
        "BNB": "binancecoin",
        "USDC": "usd-coin",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "SOL": "solana",
        "TRX": "tron",
    }

    def __init__(self, cache_duration=60):
        self.cached_data = {}
        self.last_fetch_time = {}
        self.cache_duration = cache_duration

    def fetch_real_time_data(self, coin_ticker: str):
        """
        Fetch real-time cryptocurrency data from the CoinGecko API.
        :param coin_ticker: The ticker symbol of the cryptocurrency (e.g., 'BTC', 'ETH')
        :return: A DataFrame containing the real-time data
        """
        # Convert the coin ticker to uppercase to standardize input
        coin_ticker = coin_ticker.upper()

        # Check if the coin is supported
        if coin_ticker not in self.COIN_MAPPING:
            print(f"Coin '{coin_ticker}' is not supported.")
            return pd.DataFrame()

        # Get the CoinGecko name for the coin
        coin_name = self.COIN_MAPPING[coin_ticker]
        current_time = time.time()

        # Check if cache is valid (within the cache duration)
        if coin_ticker in self.cached_data and coin_ticker in self.last_fetch_time:
            if current_time - self.last_fetch_time[coin_ticker] < self.cache_duration:
                print(f"Using cached data for {coin_ticker}...")
                return self.cached_data[coin_ticker]

        # If cache is invalid or expired, fetch new data
        print(f"Fetching new data for {coin_ticker}...")

        url = f"https://api.coingecko.com/api/v3/coins/{coin_name}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "1",  # Fetch data for the past 1 day
        }

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            if not prices or not volumes:
                print(f"No data received from CoinGecko API for {coin_ticker}.")
                return pd.DataFrame()

            # Convert to DataFrame
            price_df = pd.DataFrame(prices, columns=["timestamp", "close"])
            volume_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            df = price_df.merge(volume_df, on="timestamp")
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df[["date", "close", "volume"]]
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)

            # Generate synthetic 'open', 'high', 'low' data
            df["open"] = df["close"].shift(1).fillna(df["close"])
            df["high"] = df[["open", "close"]].max(axis=1)
            df["low"] = df[["open", "close"]].min(axis=1)

            # Reorder columns
            df = df[["date", "open", "high", "low", "close", "volume"]]

            # Cache the data and update the fetch timestamp
            self.cached_data[coin_ticker] = df
            self.last_fetch_time[coin_ticker] = current_time

            return df

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed for {coin_ticker}: {e}")
            return pd.DataFrame()
        # pylint: disable=broad-except
        except Exception as e:
            print(f"An error occurred while fetching data for {coin_ticker}: {e}")
            return pd.DataFrame()

    def get_current_price(self, coin_ticker: str, currency="usd"):
        """
        Fetch the current price of a cryptocurrency from the CoinGecko API.
        :param coin_ticker: The ticker symbol of the cryptocurrency (e.g., 'BTC', 'ETH')
        :param currency: The currency in which the price is denominated (default: 'usd')
        :return: The current price of the cryptocurrency in the specified currency
        """
        coin_ticker = coin_ticker.upper()
        if coin_ticker not in self.COIN_MAPPING:
            print(f"Coin '{coin_ticker}' is not supported.")
            return None

        coin_name = self.COIN_MAPPING[coin_ticker]
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin_name, "vs_currencies": currency}

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            price = data.get(coin_name, {}).get(currency)
            if price is None:
                print(f"Price for {coin_ticker} not found.")
            return price

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            return None
        # pylint: disable=broad-except
        except Exception as e:
            print(f"An error occurred while fetching the current price: {e}")
            return None
