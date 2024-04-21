import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

@njit
def calculate_moving_average(data, window_size):
    ma = np.empty(len(data))
    for i in range(len(data)):
        if i < window_size:
            ma[i] = np.mean(data[:i+1])
        else:
            ma[i] = np.mean(data[i-window_size+1:i+1])
    return ma

@njit
def opt(prices, data, data_ma, windows, ths):
    max_score = 0
    best_window = None
    best_threshold = None
    best_eq = np.zeros(len(data))
    best_th = np.zeros(len(data))
    fee = 0.001
    best_window_idx = 0
    for w in range(len(windows)):
        for t in range(len(ths)):
            window = windows[w]
            th = ths[t]
            ma_th = data_ma[w] * th
            diff = data - ma_th
            diff_shifted = np.roll(diff, 1)
            diff_shifted[0] = diff[0]

            crossovers = np.where((diff > 0) & (diff_shifted <= 0))[0]
            crossunders = np.where((diff < 0) & (diff_shifted >= 0))[0]

            equity = np.zeros(len(data))
            equity[0] = 1
            positions_open = False
            max_eq = 1
            max_dd = 0.1

            for i in range(1,len(data)):
                if positions_open:
                    change_p = (prices[i] - prices[i-1])/prices[i-1]
                    equity[i] = equity[i-1] * (1+change_p)
                    if max_eq < equity[i]:
                        max_eq = equity[i]
                    elif (max_eq - equity[i])/max_eq > max_dd:
                        max_dd = (max_eq - equity[i])/max_eq
                else:
                    equity[i] = equity[i-1]

                if i in crossovers and not positions_open:
                    positions_open = True
                    equity[i] = equity[i]*(1-fee)
                elif i in crossunders and positions_open:
                    positions_open = False

            if equity[-1]/max_dd > max_score:
                max_score = equity[-1]/max_dd
                best_window = window
                best_threshold = th
                best_eq = equity
                best_th = ma_th
                best_co = crossovers
                best_cu = crossunders
                best_dd = max_dd
                best_window_idx = w

    return best_window, best_window_idx, best_threshold, max_score, best_eq, best_th, best_co, best_cu, best_dd

@njit
def inf(prices, data, data_ma, th):
    fee = 0.001
    ma_th = data_ma * th
    diff = data - ma_th
    diff_shifted = np.roll(diff, 1)
    diff_shifted[0] = diff[0]

    crossovers = np.where((diff > 0) & (diff_shifted <= 0))[0]
    crossunders = np.where((diff < 0) & (diff_shifted >= 0))[0]

    equity = np.zeros(len(data))
    equity[0] = 1
    positions_open = False
    max_eq = 1
    max_dd = 0.1

    for i in range(1,len(data)):
        if positions_open:
            change_p = (prices[i] - prices[i-1])/prices[i-1]
            equity[i] = equity[i-1] * (1+change_p)
            if max_eq < equity[i]:
                max_eq = equity[i]
            elif (max_eq - equity[i])/max_eq > max_dd:
                max_dd = (max_eq - equity[i])/max_eq
        else:
            equity[i] = equity[i-1]

        if i in crossovers and not positions_open:
            positions_open = True
            equity[i] = equity[i]*(1-fee)
        elif i in crossunders and positions_open:
            positions_open = False
    
    return max_dd, equity[-1]

def rolling_walk_forward_optimization(prices, datas, data_ma_s, optimization_window, steps, windows, ths):
    n = len(prices)
    opt_results = []
    inf_results = []
    inf_total_eq = 1.0
    for start_idx in tqdm(range(1000, n - optimization_window, steps), desc="wfo"):
        data_ma = data_ma_s[:,start_idx:start_idx + optimization_window]
        data = datas[start_idx:start_idx + optimization_window]
        price = prices[start_idx:start_idx + optimization_window]

        window, window_idx, th, score, output, best_th, co, cu, best_dd = opt(price, data, data_ma, windows, ths)
        
        price = prices[start_idx + optimization_window:start_idx + optimization_window + steps]
        data = datas[start_idx + optimization_window:start_idx + optimization_window + steps]
        data_ma = data_ma_s[window_idx, start_idx + optimization_window:start_idx + optimization_window + steps]
        
        max_dd, equity = inf(price, data, data_ma, th)
        inf_total_eq *= equity
        # print('result:', equity)
        # print('out_sample_equity:', inf_total_eq)

        opt_result = (start_idx, window, th, score, output[-1])
        inf_result = (start_idx + optimization_window, equity)
        opt_results.append(opt_result)
        inf_results.append(inf_result)
    print('total_est_out_sample_equity:', inf_total_eq)
    return opt_results, inf_results

df = pd.read_csv('BTCUSDT_1h_nprice.csv', index_col='timestamp', parse_dates=True)

ma_combs = np.arange(10, 1000, 5)
thresholds = np.arange(-3.0, 3.0, 0.05)
in_window = 5000
out_window = 400

results = []
for ma_comb in ma_combs:
    results.append(calculate_moving_average(df['Signal'].values, ma_comb))

precomputed_ma = np.array(results)
opt_results, inf_results = rolling_walk_forward_optimization(df['close'].values, df['Signal'].values, precomputed_ma, in_window, out_window, ma_combs, thresholds)
print(opt_results)


