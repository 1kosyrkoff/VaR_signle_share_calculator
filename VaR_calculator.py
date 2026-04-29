import pandas as pd
import numpy as np
import requests as re
import time
import matplotlib.pyplot as plt
from arch import arch_model
from datetime import datetime, timedelta
from scipy import stats
from tqdm import tqdm
def get_tickers():
    url = f'https://iss.moex.com/iss/engines/stock/markets/shares/securities.json'
    params = {
        "iss.only" : "securities",
        "securities.columns" : "SECID"
    }
    resp = re.get(url, params=params)
    resp = resp.json()['securities']
    r = [k[0] for k in resp["data"]]
    return r


def check_for_existance(ticker, r):
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
    flow = flow.sort_values(by='ln_change')
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
    flow = flow.sort_values(by='ln_change')
    print(f'\np_value Шапиро-Уилка: {stats.shapiro(np.array(flow['ln_change'])).pvalue}')
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
    flow = flow.sort_values(by='ln_change')
    mu = flow['ln_change'].mean()
    sigma = flow['ln_change'].std()

    scenarios = np.sort(stats.norm.rvs(loc = 0, scale = 1, size = 10000))
    prices_after_1_day = []
    prices_after_10_days = []
    for i in scenarios:
        prices_after_1_day.append(np.exp((mu - (sigma ** 2) / 2) + sigma * i))
        prices_after_10_days.append(np.exp((mu - (sigma ** 2) / 2) * 10 + sigma* np.sqrt(10) * i))
    ans = {
        "VaR_95_1" : 1 - np.percentile(prices_after_1_day, 5),
        "VaR_95_10" : 1 - np.percentile(prices_after_10_days, 5),
        "VaR_99_1" : 1 - np.percentile(prices_after_1_day, 1),
        "VaR_99_10" : 1 - np.percentile(prices_after_10_days, 1)
    }

    return (ans, "Подход Монте-Карло")


def monte_carlo_garch_VaR(data):
    flow = flow_creator(data)
    mu = flow['ln_change'].mean()
    a_model = arch_model(flow['ln_change'].values * 100, vol='Garch', p = 1, q = 1, dist='normal', rescale=False)
    res = a_model.fit(disp='off', options={'ftol' : 1e-8, 'maxiter' : 500})
    om = res.params['omega'] / 10000
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']
    sigma_starting = (res.conditional_volatility[-1] / 100)** 2
    n_sims = 10000
    max_d = 10
    r_cumulative_1 = np.zeros(n_sims)
    r_cumulative_10 = np.zeros(n_sims)
    for i in range(n_sims):
        sigma2 = sigma_starting
        r_cumulative = 0.0
        for j in range(1, 11):
            z = np.random.normal()
            sigma = np.sqrt(sigma2)
            r_day = mu + sigma * z
            r_cumulative += r_day
            if j == 1:
                r_cumulative_1[i] = r_cumulative
            if j == 10:
                r_cumulative_10[i] = r_cumulative
            sigma2 = om + alpha * (r_day ** 2) + beta * sigma2
    loss_1 = 1 - np.exp(r_cumulative_1)
    loss_10 = 1 - np.exp(r_cumulative_10)
    ans = {
        "VaR_95_1"  : float(np.percentile(loss_1, 5)),
        "VaR_95_10" : float(np.percentile(loss_10, 5)),
        "VaR_99_1" : float(np.percentile(loss_1, 1)),
        "VaR_99_10" : float(np.percentile(loss_10, 1))
    }
    return (ans, "Подход Монте-Карло GARCH")


def flow_creator(data):
    flow = data[['TRADEDATE', 'CLOSE_ADJ']]
    flow['ln_change'] = np.log(flow['CLOSE_ADJ']/ flow['CLOSE_ADJ'].shift(1))
    flow = flow.dropna()
    flow = flow.sort_values(by='TRADEDATE')
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


def plotter(data, ticker):
    flow = flow_creator(data)
    r_s = flow['ln_change'].values
    mu = r_s.mean()
    sigma = r_s.std()
    norm_rasp = np.linspace(r_s.min(), r_s.max(), 100)
    plt.figure(figsize=(8, 6))
    plt.hist(r_s, bins=50, density=True, label='Логдоходности', alpha=0.5)
    plt.plot(norm_rasp, stats.norm.pdf(norm_rasp, r_s.mean(), r_s.std()), label='Нормальное распределение')
    plt.legend()
    plt.title(ticker)
    plt.show()


def VaRcalc():
    ticker, n_days = get_ticker_and_days()
    r = get_tickers()
    z = check_for_existance(ticker, r)
    if z == 1:
        ticker_data = return_response_moex(ticker, n_days)
        hist_var = historical_VaR(ticker_data)
        delta_norm = delta_normal_VaR(ticker_data)
        monte_calro = monte_carlo_VaR(ticker_data)
        monte_carlo_garch = monte_carlo_garch_VaR(ticker_data)
        print()
        print(ticker)
        beau_printer(*hist_var)
        beau_printer(*delta_norm)
        beau_printer(*monte_calro)
        beau_printer(*monte_carlo_garch)
        plotter(ticker_data, ticker)
    else:
        print("Тикер отсутствует на бирже")
    return 0


def multiple_VaRcalc():
    ts = []
    z = 0
    x = int(input("Кол-во тикеров\n\r"))
    n_days = 501
    r = get_tickers()
    for i in range(x):
        temp = str(input(f"Введите тикер номер {i+1}\n\r")).upper()
        ts.append(temp)
        checker = check_for_existance(temp, r)
        if checker == 0:
            print("Тикер отсутствует на бирже")
            break
        else:
            z += checker
    print("\n\n")
    if z == x:
        for ticker in tqdm(ts, desc="Расчет всех сценариев"):
            ticker_data = return_response_moex(ticker, n_days)
            hist_var = historical_VaR(ticker_data)
            delta_norm = delta_normal_VaR(ticker_data)
            monte_calro = monte_carlo_VaR(ticker_data)
            monte_carlo_garch = monte_carlo_garch_VaR(ticker_data)
            print(ticker)
            beau_printer(*hist_var)
            beau_printer(*delta_norm)
            beau_printer(*monte_calro)
            beau_printer(*monte_carlo_garch)
            print("\n\n")
    return 0


opt = int(input("Введите 1 для VaR одной акции\nВведите 2 для VaR списка акций\n\r"))
if opt == 1:
    VaRcalc()
elif opt == 2:
    multiple_VaRcalc()
else:
    print("Некорректный ввод")
