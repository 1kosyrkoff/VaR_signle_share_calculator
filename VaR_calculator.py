import pandas as pd
import numpy as np
import requests as re
import time
from datetime import datetime, timedelta
from scipy import stats
from tqdm import tqdm
def check_for_existance(ticker):
    url = f'https://iss.moex.com/iss/engines/stock/markets/shares/securities.json'
    params = {
        "iss.only" : "securuties",
        "securities.columns" : "SECID"
    }
    resp = re.get(url, params=params)
    resp = resp.json()['securities']
    r = [k[0] for k in resp["data"]]
    if ticker in r:
        ans = 1
    else:
        ans = 0
    return ans


def get_ticker_and_days():
    ticker = str(input("Введите тикер акции \n\r")).upper()

    n_days = int(input("Введите кол-во дней истории (рекомендовано: 501) \n\r"))
    return ticker, n_days


def return_response_moex(ticker, n_days):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days = n_days)
    url = f'https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json'
    params = {
        'from' : start_date,
        'till' : end_date,
        'limit' : 1000,
        'iss.only' : 'history'
    }
    resp = re.get(url, params=params)
    data = resp.json()['history']
    data = pd.DataFrame(data['data'], columns=data['columns'])

    time.sleep(1.1)
    url_div = f'https://iss.moex.com/iss/securities/{ticker}/dividends.json'
    params_div = {
        'from' : start_date,
        'till' : end_date,
        'limit' : 1000,
        'iss.only': 'dividends'}
    resp_div = re.get(url_div, params=params_div)
    data_div = resp_div.json()['dividends']
    data_div = pd.DataFrame(data_div['data'], columns=data_div['columns'])

    data['CLOSE_ADJ'] = data['CLOSE'].copy()
    data['TRADEDATE'] = pd.to_datetime(data['TRADEDATE'])
    if not data_div.empty:
        data_div['registryclosedate'] = pd.to_datetime(data_div['registryclosedate'])
        data_div = data_div.sort_values(by='registryclosedate', ascending=False)
        for i, row in data_div.iterrows():
            div_date = row['registryclosedate']
            div_value = row['value']
            mask = data['TRADEDATE'] >= div_date
            if not mask.any():
                continue
            price_ex = data.loc[mask, 'CLOSE'].iloc[0]
            if price_ex <= div_value:
                continue
            adj_factor = (price_ex - div_value) / price_ex
            data.loc[data['TRADEDATE'] < div_date, "CLOSE_ADJ"] *= adj_factor
    return data


def historical_VaR(data):
    def calc_VaR(series, conf, days):
        conf_int = 1 - conf
        picked_value_i = int(np.ceil(conf * len(series))) - 1
        return series['ln_change'].iloc[picked_value_i] * np.sqrt(days)

    flow = flow_creator(data)
    ans = { 
        "VaR_95_1" : float(calc_VaR(flow, 0.05, 1)),
        "VaR_95_10" : float(calc_VaR(flow, 0.05, 10)),
        "VaR_99_1" : float(calc_VaR(flow, 0.01, 1)),
        "VaR_99_10" : float(calc_VaR(flow, 0.01, 10))
    }
    return (ans, "Исторический подход")


def delta_normal_VaR(data):
    def calc_VaR_dn(sigma, conf, days):
        q = stats.norm.ppf(1 - conf)
        return q * np.sqrt(days) * sigma

    flow = flow_creator(data)
    print(f'p_value Шапиро-Уилка: {stats.shapiro(np.array(flow['ln_change'])).pvalue}')
    sigma = flow['ln_change'].std()
    ans = {
        "VaR_95_1" : float(calc_VaR_dn(sigma, 0.05, 1)),
        "VaR_95_10" : float(calc_VaR_dn(sigma, 0.05, 10)),
        "VaR_99_1" : float(calc_VaR_dn(sigma, 0.01, 1)),
        "VaR_99_10" : float(calc_VaR_dn(sigma, 0.01, 10))
    }
    return (ans, "Дельта-нормальный подход")


def monte_carlo_VaR(data):
    flow = flow_creator(data)
    mu = flow['ln_change'].mean()
    sigma = flow['ln_change'].std()

    scenarios = np.sort(stats.norm.rvs(loc = 0, scale = 1, size = 10000))
    prices_after_1_day = []
    prices_after_10_days = []
    for i in tqdm(scenarios, desc="Расчет сценариев"):
        prices_after_1_day.append(np.exp((mu - (sigma ** 2) / 2) + sigma * i))
        prices_after_10_days.append(np.exp((mu - (sigma ** 2) / 2) * 10 + sigma* np.sqrt(10) * i))
    ans = {
        "VaR_95_1" : 1 - np.percentile(prices_after_1_day, 5),
        "VaR_95_10" : 1 - np.percentile(prices_after_10_days, 5),
        "VaR_99_1" : 1 - np.percentile(prices_after_1_day, 1),
        "VaR_99_10" : 1 - np.percentile(prices_after_10_days, 1)
    }

    return (ans, "Подход Монте-Карло")


def flow_creator(data):
    flow = data[['TRADEDATE', 'CLOSE_ADJ']]
    flow['ln_change'] = np.log(flow['CLOSE_ADJ']/ flow['CLOSE_ADJ'].shift(1))
    flow = flow.dropna()
    flow = flow.sort_values(by='ln_change')
    return flow


def beau_printer(ans, name):
    print(f"{name}")
    print("-" * 22)
    print("Значения в %")
    for key, value in ans.items():
        print(f"{key} : {100 * abs(value):.6f}")
    print("Справка:\nVaR_(%)_(кол-во дней)")
    print("-" * 22)
    print()
    return 0


def VaRcalc():
    ticker, n_days = get_ticker_and_days()
    z = check_for_existance(ticker)
    if z == 1:
        ticker_data = return_response_moex(ticker, n_days)
        hist_var = historical_VaR(ticker_data)
        delta_norm = delta_normal_VaR(ticker_data)
        monte_calro = monte_carlo_VaR(ticker_data)
        print()
        beau_printer(*hist_var)
        beau_printer(*delta_norm)
        beau_printer(*monte_calro)
    else:
        print("Тикер отсутствует на бирже")
    return 0

VaRcalc()
