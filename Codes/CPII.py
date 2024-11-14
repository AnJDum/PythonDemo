# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

# Input:
#     risk_multiplier: Risk multiplier
#     equity_num: Number of high-risk assets
#     bond_num: Number of low-risk assets
#     return_data: Returns for each asset, arranged as high-risk assets followed by low-risk assets
#     rf: Risk-free rate
# Output: weight as the weights for each asset, with equal weight allocation within high/low-risk asset classes

def Model_CPPI(risk_multiplier, equity_num, bond_num, return_data, rf, ann, guarantee_rate=1, risk_trading_fee_rate=0):
    # Define CPPI parameters

    init_nav = 1  # Initial capital
    adj_period = 1  # Rebalancing period
    eq_ratio = 0.35

    # Fixed allocation for fixed-income assets
    fix_asset = init_nav * 0.36
    free_asset = init_nav * 0.64

    # Generate returns for fixed-income assets
    if bond_num == 0:
        rf_return = np.zeros(return_data.shape[0])
    else:
        # Separate fixed income and bonds (not averaged). Fixed-income (fi) and bonds (rf) are separated in the following lines.
        rf_return = return_data.iloc[:, 1].values
        fi_return = return_data.iloc[:, 2].values

    if equity_num == 0:
        risk_return = np.zeros(return_data.shape[0])
    else:
        risk_return = np.mean(return_data.iloc[:, 0:equity_num].values, 1)  # Return for high-risk assets

    # Calculate theoretical bond floor
    n1 = len(return_data)
    min_pv_asset = np.zeros((n1 + 1, 1))
    min_pv_asset[n1] = 1
    for i in range(n1):
        min_pv_asset[i] = np.exp(-rf * (n1 - i + 1) / ann)

    # CPPI Strategy
    trading_sim = n1
    risk_asset = np.zeros((trading_sim + 1, 1))  # High-risk assets (equity)
    rf_asset = np.zeros((trading_sim + 1, 1))  # Risk-free assets (bond)
    fi_asset = np.zeros((trading_sim + 1, 1))  # Fixed-income non-standard assets
    floor = np.zeros((trading_sim + 1, 1))  # Value floor
    nav = np.zeros((trading_sim + 1, 1))  # Total portfolio value
    nav_weight = np.zeros((trading_sim + 1, 1))  # NAV for calculating weights

    weight_raw = np.zeros((trading_sim + 1, 3))
    nav[0] = init_nav
    nav_weight[0] = init_nav

    # Day 1
    floor[0] = guarantee_rate * min_pv_asset[0]  # Floor on Day 1
    risk_asset[0] = np.minimum(np.maximum(0, risk_multiplier[0] * (nav[0] - floor[0])), eq_ratio)  # Position in high-risk assets on Day 1
    fi_asset[0] = fix_asset
    rf_asset[0] = nav[0] - risk_asset[0] - fi_asset[0]  # Position in risk-free assets
    risk_asset[0] = risk_asset[0] * (1 - risk_trading_fee_rate)  # Deduct trading fee

    # Day 2 to last day

    # Generate month-end indices
    offset = MonthEnd()
    month_end = return_data.groupby(offset.rollforward).apply(lambda t: t[t.index == t.index.max()])
    end_date_index = []
    i = 0
    for t in range(len(return_data)):
        if month_end.index[i][1] == return_data.index[t]:
            end_date_index.append(t + 1)
            i += 1

    for t in range(1, trading_sim + 1):
        # Daily NAV calculation
        floor[t] = guarantee_rate * min_pv_asset[t]  # Floor value
        risk_asset[t] = (1 + risk_return[t - 1]) * risk_asset[t - 1]
        fi_asset[t] = (1 + fi_return[t - 1]) * fi_asset[t - 1]
        rf_asset[t] = (1 + rf_return[t - 1]) * rf_asset[t - 1]
        nav[t] = risk_asset[t] + rf_asset[t] + fi_asset[t]
        nav_weight[t] = risk_asset[t] + rf_asset[t] + fi_asset[t]

        # Rebalance at month-end
        if t in end_date_index:
            a = 1
            risk_asset_b4_adj = risk_asset[t]
            risk_asset[t] = np.minimum(
                np.maximum(0, risk_multiplier[t] * (nav_weight[t] - floor[t])), eq_ratio * nav_weight[t])  # High-risk asset allocation
            rf_asset[t] = nav_weight[t] - risk_asset[t] - fi_asset[t]  # Risk-free asset allocation
            risk_asset[t] = risk_asset[t] - abs(
                risk_asset_b4_adj - risk_asset[t]) * risk_trading_fee_rate  # Trading fee, applied on both buy and sell
            nav_weight[t] = risk_asset[t] + rf_asset[t] + fi_asset[t]

            # Check for forced liquidation
            if risk_asset[t] <= 0:
                rf_asset[t] = nav_weight[t] - risk_asset[t] * risk_trading_fee_rate - fi_asset[t]
                risk_asset[t] = 0
                nav_weight[t] = risk_asset[t] + rf_asset[t] + fi_asset[t]

            # Equity position cap at 20%, transfer excess to bonds
            if risk_asset[t] / (risk_asset[t] + rf_asset[t] + fi_asset[t]) > eq_ratio:
                temp = risk_asset[t]
                risk_asset[t] = (risk_asset[t] + rf_asset[t] + fi_asset[t]) * eq_ratio
                rf_asset[t] = rf_asset[t] + temp - risk_asset[t]

            nav_weight[t] = risk_asset[t] + rf_asset[t] + fi_asset[t]

    weight_raw[:, 0] = (risk_asset / (risk_asset + rf_asset + fi_asset)).flatten()
    weight_raw[:, 1] = (rf_asset / (risk_asset + rf_asset + fi_asset)).flatten()
    weight_raw[:, 2] = (fi_asset / (risk_asset + rf_asset + fi_asset)).flatten()
    weight_raw = weight_raw[end_date_index, :]

    weight = np.zeros((len(end_date_index) + 1, equity_num + bond_num))
    weight[0, 0] = risk_asset[0] / nav_weight[0]
    weight[0, 1] = rf_asset[0] / nav_weight[0]
    weight[0, 2] = fi_asset[0] / nav_weight[0]
    weight[1:, 0:equity_num] = np.repeat(np.expand_dims(weight_raw[:, 0] / equity_num, axis=1), equity_num, axis=1)
    weight[1:, 1:2] = np.repeat(np.expand_dims(weight_raw[:, 1] / 1, axis=1), 1, axis=1)
    weight[1:, 2:3] = np.repeat(np.expand_dims(weight_raw[:, 2] / 1, axis=1), 1, axis=1)
    a = 1
    return weight
