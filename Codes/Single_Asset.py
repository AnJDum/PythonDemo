import pandas as pd
import numpy as np

class Single_Asset:
    def __init__(self, ann: int, rf=0.0, data=None):
        """
        Initialize a backtester for a single asset
        Processes a closing price series and outputs several statistics
        Cannot apply position allocations to the closing price series
        :param int ann: number of periods used to annualize statistics, e.g., 250 for daily closing prices, 52 for weekly,
        12 for monthly, etc.
        :param float rf: risk-free rate, default 0
        :param pd.DataFrame data: closing price series, allowing use of this backtester after other
        Python programs without loading from a local Excel file
        """
        self.ann = ann
        self.rf = rf
        self.input_path = self.output_path = None
        self.data = data
        self.backtest_results = dict()

    def load_sheet_from_file(self, input_path: str, sheet_name='Sheet1'):
        """
        Load closing price series data from a local Excel file
        :param str input_path: file path of the Excel file
        :param str sheet_name: name of the sheet containing closing price series, default 'Sheet1'
        """
        self.input_path = input_path
        self.data = pd.read_excel(self.input_path, sheet_name=sheet_name, index_col=0)
        self.data.sort_index(ascending=True, inplace=True)
        # If self.data is not sorted, slicing and backtesting below may lead to errors; if already sorted,
        # this operation has no effect.

    def slice(self, start_date=None, end_date=None):
        """
        Slice the closing price series data based on desired start and end dates
        :param str start_date: desired start date
        :param str end_date: desired end date
        """
        self.data = self.data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

    def backtest_series(self, nav_series: pd.Series, annualize: bool):
        """
        Backtest the given closing price series for the specified period
        This method can be used to backtest both the entire period and individual years by slicing nav_series accordingly.
        :param pd.Series nav_series: closing price series used to calculate statistics
        :param bool annualize: whether to annualize returns
        :return pd.DataFrame: a DataFrame of statistics
        """
        ret_series = nav_series.pct_change()

        # --------------------------------------------------------------------------------------------------------------
        # Calculate return and standard deviation
        holding_period_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
        annualized_return = (holding_period_return + 1) ** (self.ann / (len(ret_series) - 1)) - 1 if annualize else holding_period_return
        annualized_stdev = np.nanstd(ret_series, ddof=1) * np.sqrt(self.ann)

        # --------------------------------------------------------------------------------------------------------------
        # Calculate maximum drawdown (MDD) and performance ratios
        mdd, mdd_start, mdd_formation = self.mdd(nav_series)
        # Calculate the number of trading days from mdd_start to mdd_formation
        formation_period = len(nav_series.loc[mdd_start : mdd_formation]) - 1

        sharpe = (annualized_return - self.rf) / annualized_stdev
        calmar = (annualized_return - self.rf) / mdd if mdd > 0 else np.nan

        # --------------------------------------------------------------------------------------------------------------
        # Store results in a DataFrame
        df = pd.DataFrame(
            {'Period Return': [holding_period_return], 'Annualized Return': [annualized_return],
             'Annualized Volatility': [annualized_stdev], 'Maximum Drawdown': [mdd], 'Sharpe Ratio': [sharpe],
             'Calmar Ratio': [calmar], 'MDD Start Date': [mdd_start], 'MDD Formation Period (Trading Days)': [formation_period],
             'MDD Formation Date': [mdd_formation], 'MDD Recovery Period (Trading Days)': [np.nan], 'MDD Recovery Date': [np.nan]
             })

        return df

    def backtest(self, asset_name: str):
        """
        Run backtest on the selected asset
        :param str asset_name: name of the asset, used to locate the column among multiple assets
        """
        if asset_name not in self.data.columns:
            raise ValueError('Invalid asset name')
        nav_series = self.data[asset_name].dropna()
        # Since self.data can contain multiple assets, there may be empty cells

        df_list = []

        # --------------------------------------------------------------------------------------------------------------
        # Backtest the entire period
        df_all = self.backtest_series(nav_series, annualize=True)
        # ----------- Update ------------
        # Add the number of trading days from MDD formation to MDD recovery
        try:
            df_all['MDD Recovery Date'] = nav_series.loc[
                (nav_series >= nav_series.loc[df_all['MDD Start Date']][0])
                & (pd.to_datetime(nav_series.index) > pd.to_datetime(df_all['MDD Start Date'][0]))].index[0].date()
            # Count the number of dates in NAV series between two timestamps
            recover_period = len(nav_series.loc[(pd.to_datetime(nav_series.index) > pd.to_datetime(df_all['MDD Formation Date'][0]))
                                & (pd.to_datetime(nav_series.index) <= pd.to_datetime(df_all['MDD Recovery Date'][0]))])
            df_all['MDD Recovery Period (Trading Days)'] = recover_period
            
        except:
            df_all['MDD Recovery Date'] = 'Not Recovered'
            df_all['MDD Recovery Period (Trading Days)'] = 'Not Recovered'
            
        df_all.index = ['Overall Performance']
        df_list.append(df_all)       

        # --------------------------------------------------------------------------------------------------------------
        # Backtest by year
        years = list(set(nav_series.index.year))
        years.sort()
        for idx, year in enumerate(years):
            nav_series_by_year = nav_series.loc[nav_series.index.year == year]
            if idx == 0:
                # If the first year only involves one data point, it serves as the opening price for the next year's series
                if len(nav_series_by_year) == 1:
                    continue
            else:
                last_year_close = pd.Series([nav_series.loc[nav_series.index.year == years[idx - 1]].iloc[-1]])
                last_year_close.index = [nav_series.loc[nav_series.index.year == years[idx - 1]].index[-1]]
                last_year_close.name = nav_series_by_year.name
                nav_series_by_year = last_year_close.append(nav_series_by_year)
            df_by_year = self.backtest_series(nav_series_by_year, annualize=False)
            try:
                df_by_year['MDD Recovery Date'] = nav_series.loc[
                    (nav_series >= nav_series.loc[df_by_year['MDD Start Date']][0])
                    & (pd.to_datetime(nav_series.index) > pd.to_datetime(df_by_year['MDD Start Date'][0]))].index[0].date()
                recover_period = len(nav_series.loc[(pd.to_datetime(nav_series.index) > pd.to_datetime(df_by_year['MDD Formation Date'][0]))
                                & (pd.to_datetime(nav_series.index) <= pd.to_datetime(df_by_year['MDD Recovery Date'][0]))])
                df_by_year['MDD Recovery Period (Trading Days)'] = recover_period
            except:
                df_by_year['MDD Recovery Date'] = 'Not Recovered'
                df_by_year['MDD Recovery Period (Trading Days)'] = 'Not Recovered'
            df_by_year.index = [year]
            df_list.append(df_by_year)

        # --------------------------------------------------------------------------------------------------------------
        # Concatenate results into one holistic DataFrame
        df = pd.concat(df_list)
        self.backtest_results[asset_name] = df

    def mdd(self, nav_series: pd.Series):
        """
        Calculate maximum drawdown (MDD) using the given price series
        :param pd.Series nav_series: price series used to calculate MDD statistics
        :return: statistics
        """
        dd = nav_series.div(nav_series.cummax()).sub(1)
        # NAV divided by its cumulative maximum, then subtracted by 1, gives the drawdown series
        mdd, formation = dd.min(), dd.idxmin()
        formation = formation.date()
        start = nav_series.loc[:formation].idxmax()
        start = start.date()
        return -mdd, start, formation

    def output(self, output_path: str, asset_name_list: list):
        """
        Save results as an Excel file
        :param str output_path: desired file path of the Excel file containing backtest results
        :param list asset_name_list: list of names of assets for which backtest results are to be output
        """
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        for asset in asset_name_list:
            if asset not in self.backtest_results.keys():
                print('Invalid asset %s, either no data or has not been backtested.' % asset)
            else:
                self.backtest_results[asset].to_excel(writer, sheet_name=asset)
        writer.save()

if __name__ == '__main__':
    a = Single_Asset(ann=250, rf=0)
    input_path = r'..\Test\05_Leverage_and_Short_Selling\data.xlsx'
    a.load_sheet_from_file(input_path=input_path, sheet_name='Data')
    start_date = None
    end_date = None
    a.slice(start_date, end_date)
    for asset in a.data.columns:
        a.backtest(asset)
    output_path = r'..\Test\05_Leverage_and_Short_Selling\Single_Asset_Backtest_Results_test.xlsx'
    a.output(output_path=output_path, asset_name_list=list(a.data.columns))
