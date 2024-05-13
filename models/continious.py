#!/usr/bin/env python3

import os
import sqlite3
from datetime import datetime

import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException

# Load environment variables
load_dotenv()

# API configuration
API_URL = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
API_KEY = os.getenv("COIN_MARKET_API")


def fetch_cryptocurrency_data(symbol="BTC", currency="USD"):
    """Fetch cryptocurrency data from the CoinMarketCap API."""
    params = {"symbol": symbol, "convert": currency}
    headers = {"Accept": "application/json", "X-CMC_PRO_API_KEY": API_KEY}
    try:
        with requests.Session() as session:
            response = session.get(API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data["data"][symbol][0]["quote"][currency]
    except RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def create_db_connection(db_file="btc_data.db"):
    """Create and return a database connection and cursor."""
    conn = sqlite3.connect(db_file)
    return conn


def create_table(conn):
    """Create the database table if it does not already exist."""
    with conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS price (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                percent_change_1h REAL,
                percent_change_24h REAL,
                percent_change_7d REAL,
                percent_change_30d REAL,
                percent_change_60d REAL,
                percent_change_90d REAL,
                price REAL,
                volume_24h REAL,
                volume_change_24h REAL,
                market_cap REAL,
                market_cap_dominance REAL,
                fully_diluted_market_cap REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )


def insert_data(conn, data):
    """Insert data into the database."""
    with conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO price (
                percent_change_1h, percent_change_24h, percent_change_7d,
                percent_change_30d, percent_change_60d, percent_change_90d,
                price, volume_24h, volume_change_24h,
                market_cap, market_cap_dominance, fully_diluted_market_cap,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["percent_change_1h"],
                data["percent_change_24h"],
                data["percent_change_7d"],
                data["percent_change_30d"],
                data["percent_change_60d"],
                data["percent_change_90d"],
                data["price"],
                data["volume_24h"],
                data["volume_change_24h"],
                data["market_cap"],
                data["market_cap_dominance"],
                data["fully_diluted_market_cap"],
                datetime.strptime(data["last_updated"], "%Y-%m-%dT%H:%M:%S.%fZ"),
            ),
        )


def fetch_all_data(conn):
    """Fetch all data from the price table."""
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM price")
        rows = cur.fetchall()

        # Optional: convert to list of dictionaries for easier handling
        columns = [column[0] for column in cur.description]
        data = [dict(zip(columns, row)) for row in rows]
        return data


def main():
    btc_data = fetch_cryptocurrency_data()
    if btc_data:
        conn = create_db_connection()
        create_table(conn)
        insert_data(conn, btc_data)
        conn.close()
        print(f"Data inserted: {btc_data}")


if __name__ == "__main__":
    main()
