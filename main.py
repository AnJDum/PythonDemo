from Codes.Portfolio import Portfolio 
import pandas as pd
import numpy as np
import os
import openpyxl
from Codes.CPII import Model_CPPI
from pandas.tseries.offsets import MonthEnd

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # Set Backtesting Parameters
    ann = 250   # Data frequency, representing daily data; if monthly, ann = 12
    rf = 0
    equity_num = 1  # Number of equity assets
    bond_num = 2  # Number of bond assets
    start_date = None  # Format: '2015-03-04'
    end_date = None  # Format: '2020-07-05'
    high_risk_name_list = ['CSI 800 Index']  # Enter high-risk asset names in Excel column order
    high_risk_fee_rate = 0  # Transaction fee
    low_risk_name_list = ['ChinaBond Composite Index', 'Non-standard']  # Enter low-risk asset names in Excel column order
    low_risk_fee_rate = 0
    input_path = r'Data/input.xlsx'
    output_path_portfolio = r'Output/CPPI_Model_Backtesting_Results.xlsx'

    # Set Model Parameters
    guarantee_rate = 0.9  # Capital protection rate
    m = 2  # Risk multiplier

    # Load Data and Calculate Returns
    df = pd.read_excel(input_path, sheet_name='Data', index_col=[0])  # Read closing prices
    df = df.loc[start_date:end_date, :]

    retall = pd.DataFrame(index=df.index)
    retall[[x + '_Return' for x in df.columns]] = df / df.shift(1) - 1  # Calculate asset returns
    retall = retall.dropna()
    [datenum, assetnum] = retall.shape  # Calculate number of days and number of assets

    # Calculate Allocation Ratios for Each Asset
    risk_multiplier = m * np.ones(len(retall) + 1)  # Risk multiplier per period, column vector with length equal to the required number of periods
    weight = pd.DataFrame(Model_CPPI(risk_multiplier, equity_num, bond_num, retall, rf, ann, guarantee_rate, high_risk_fee_rate), columns=df.columns)  # weight_CPPI is the output weight result
    weight.columns = df.columns

    offset = MonthEnd()
    month_end = df.groupby(offset.rollforward).apply(lambda t: t[t.index == t.index.max()])
    df_index = pd.DataFrame()
    month_end.to_csv('m.csv')
    month_end = pd.read_csv('m.csv')
    col_name = month_end.columns[1]
    month_end[col_name] = pd.to_datetime(month_end[col_name], format='%Y-%m-%d')
    weight.index = month_end[col_name]

    # Remove Previously Calculated Weights
    try:
        sExcelFile = input_path
        wb = openpyxl.load_workbook(sExcelFile)
        ws = wb["Weights"]
        wb.remove(ws)
        wb.save(sExcelFile)
    except:
        pass

    # Import Weights Calculated in This Run
    with pd.ExcelWriter(input_path, mode='a', engine="openpyxl") as writer:
        weight.to_excel(writer, sheet_name='Weights')

    # Portfolio Backtesting
    portfolio = Portfolio(ann=ann, rf=rf)
    portfolio.load_sheets_from_file(input_path=input_path, data_sheet_name='Data', weight_sheet_name='Weights')
    portfolio.load_fee_rates(high_risk_name_list=high_risk_name_list, high_risk_fee_rate=high_risk_fee_rate,
                             low_risk_name_list=low_risk_name_list, low_risk_fee_rate=low_risk_fee_rate)
    portfolio.slice(start_date, end_date)
    portfolio.generate_nav()
    portfolio.backtest()
    portfolio.output(output_path=output_path_portfolio)
