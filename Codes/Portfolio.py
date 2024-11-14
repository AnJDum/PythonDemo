import pandas as pd
from scipy.optimize import newton
from .Single_Asset import Single_Asset
import matplotlib.pyplot as plt
import warnings

pd.set_option('display.max_rows', 5000)
warnings.filterwarnings('ignore')

class Portfolio:
    def __init__(self, ann: int, rf: float, data=None, weight=None):
        """
        Initialize a backtester for a portfolio
        :param int ann: number of days used to annualize statistics, e.g. 250 or 252
        :param float rf: risk-free rate
        :param pd.DataFrame data: closing price series, so that you can also use this backtester after some other
        Python programs without loading a local Excel file
        :param pd.DataFrame weight: asset weight series, so that you can also use this backtester after some other
        Python programs without loading a local Excel file
        """
        self.ann = ann
        self.rf = rf

        self.input_path = self.output_path = None
        self.data = data
        self.weight = weight

        self.high_risk_name_list = self.high_risk_fee_rate = self.low_risk_name_list = self.low_risk_fee_rate = None

        self.backtest_results = dict()

    def load_sheets_from_file(self, input_path: str, data_sheet_name='Data', weight_sheet_name='Weights'):
        """
        Load closing price series and weight series data from a local Excel file
        :param str input_path: file path of the Excel file
        :param str data_sheet_name: name of the sheet containing closing price series, default 'Data'
        :param str weight_sheet_name: name of the sheet containing weight series, default 'Weights'
        """
        self.input_path = input_path
        self.data = pd.read_excel(self.input_path, sheet_name=data_sheet_name, index_col=0)
        self.data.sort_index(ascending=True, inplace=True)
        self.weight = pd.read_excel(self.input_path, sheet_name=weight_sheet_name, index_col=0)
        self.weight.sort_index(ascending=True, inplace=True)
        self.data = self.data[self.weight.columns]
        # only keep columns in self.data that have weight information; if not loading closing price and weight
        # series from a local Excel file, the user must ensure alignment before initializing a backtester

    def load_fee_rates(self, high_risk_name_list=None, high_risk_fee_rate=None, low_risk_name_list=None,
                       low_risk_fee_rate=None):
        """
        Specify high and low risk assets along with applicable fee rates for each category
        :param list high_risk_name_list: list of names of high-risk assets
        :param float high_risk_fee_rate: fee rate for high-risk assets
        :param list low_risk_name_list: list of names of low-risk assets (excluding cash)
        :param float low_risk_fee_rate: fee rate for low-risk assets (excluding cash)
        """
        duplicate_assets = list(set(high_risk_name_list) & set(low_risk_name_list))  # Find overlapping assets
        if len(duplicate_assets) > 0:
            raise ValueError('Assets found in both high and low risk asset name lists: %s.' %
                             ', '.join(duplicate_assets))

        unspecified_assets = list(set(self.weight.columns) - (set(high_risk_name_list) | set(low_risk_name_list)))
        if len(unspecified_assets) > 0:
            raise ValueError('Risk level unspecified for assets: %s.' % ', '.join(unspecified_assets))

        self.high_risk_name_list, self.high_risk_fee_rate = high_risk_name_list, high_risk_fee_rate
        self.low_risk_name_list, self.low_risk_fee_rate = low_risk_name_list, low_risk_fee_rate

    def slice(self, start_date=None, end_date=None):
        """
        Slice the closing price series data based on the desired start and end dates and process both closing price and weight
        dataframes.
        This method does more than merely slicing but is named "slice" to mirror the method in Single_Asset.
        :param str start_date: desired start date
        :param str end_date: desired end date
        """
        # --------------------------------------------------------------------------------------------------------------
        # Step 1: General slicing

        self.data = self.data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
        # Remaining slice of NAV series falls between start and end dates inclusive. It does NOT necessarily contain
        # start or end dates depending on the original input.

        self.weight = self.weight.loc[self.data.index[0]: self.data.index[-1]]
        # Any weight info outside of the NAV time span is discarded.
        # --------------------------------------------------------------------------------------------------------------
        # Step 2: Adjust the weight dataframe
        # Backtesting imposes a weighting structure on the NAV series, so we adjust the weight dataframe to ensure its
        # index is a subset of the NAV dataframe index.

        date_adjusted_weight = pd.DataFrame(columns=self.weight.columns)

        for date in self.weight.index:  # Iterate through all dates in the weight dataframe

            # Check whether the weight value is valid.
            for column_name in self.weight.columns:
                if abs(self.weight.loc[date, column_name]) > 1:
                    raise ValueError("Unexpected weight value: the absolute value of asset's weight is greater than 1.")

            if date in self.data.index:    # If date exists in the NAV dataframe, it can be safely kept
                date_adjusted_weight.loc[date] = self.weight.loc[date]

            else:   # If date does NOT exist in the NAV dataframe...
                nearest_closing_date = pd.Series(self.data.index, index=self.data.index).loc[:date].iloc[-1]

                if nearest_closing_date in self.weight.index:
                    # If weight information for the nearest closing date is already specified...
                    continue    # simply discard it

                else:
                    # If weight information for the nearest closing date is NOT specified...
                    date_adjusted_weight.loc[nearest_closing_date] = self.weight.loc[date]
                    # Move it to the nearest closing date, useful for scenarios like monthly backtesting where weight
                    # series are indexed by the last calendar date of each month instead of the last trading date

        # --------------------------------------------------------------------------------------------------------------
        # Step 3: Final processing
        # Now that the weight dataframe's index is a subset of the NAV dataframe's index, we slice the NAV dataframe
        # again to ensure index alignment on the initial start date, though not necessarily the end date since backtesting
        # can extend as long as the NAV series allows after the last rebalancing.
        date_adjusted_weight.sort_index(ascending=True, inplace=True)
        self.weight = date_adjusted_weight
        self.data = self.data.loc[self.weight.index[0]:]

    def calculate_fee(self, sb: pd.Series, sa: pd.Series, f: pd.Series, pa: pd.Series) -> float:
        """
        Calculate the fee incurred for a given rebalancing
        :param pd.Series sb: shares right before rebalancing
        :param pd.Series sa: shares right after rebalancing
        :param pd.Series f: fee rate vector
        :param pd.Series pa: closing prices right after rebalancing, i.e., the prices used for rebalancing
        :return float: fee incurred for the given rebalancing
        """
        return sum(abs(sb - sa) * f * pa)

    def generate_nav(self):
        """
        Generate NAV series of the portfolio based on closing prices and weight data
        """
        # --------------------------------------------------------------------------------------------------------------
        # Step 1: Initialize dataframes and series to store various results
        data = self.data / self.data.iloc[0]
        # Normalize initial prices to 1 for floating-point precision
        input_weight = self.weight

        portfolio_stats = pd.DataFrame(index=data.index)
        shares = pd.DataFrame(index=data.index, columns=data.columns)
        actual_weight = pd.DataFrame(index=data.index, columns=data.columns)
        turnover_ratio = pd.DataFrame(index=data.index, columns=data.columns)
        fee = pd.Series(index=data.index)
        fee_rate = pd.Series(index=data.columns)
        nav = pd.Series(index=data.index)

        # --------------------------------------------------------------------------------------------------------------
        # Step 2: Fill in the fee rate vector, then define a function to solve for NAV after rebalancing
        for asset in fee_rate.index:
            if asset in self.high_risk_name_list:
                fee_rate.loc[asset] = self.high_risk_fee_rate
            elif asset in self.low_risk_name_list:
                fee_rate.loc[asset] = self.low_risk_fee_rate

        def nav_equation(x: float, sb: pd.Series, wa: pd.Series, pa: pd.Series, f: pd.Series, navb: float):
            """
            Solve for NAV right after rebalancing using the fundamental relationship that the change in NAV should equal
            the fee incurred; total fee is the sum over all assets; each asset’s fee is calculated as its fee rate
            times the rebalancing price times the absolute change in shares.
            :param pd.Series sb: shares right before rebalancing
            :param pd.Series wa: weight right after rebalancing
            :param pd.Series pa: closing prices right after rebalancing
            :param pd.Series f: fee rate vector
            :param float navb: NAV right before rebalancing
            :return float: return value of the root-solving function, i.e., x is the solution when this function
            returns 0
            """
            return self.calculate_fee(sb=sb, sa=(wa/pa)*x, f=f, pa=pa) - navb + x

        # --------------------------------------------------------------------------------------------------------------
        # Step 3: Backtest the portfolio, calculate detailed statistics
        for idx, date in enumerate(data.index):

            if idx == 0:    # For the start date...
                actual_weight.loc[date] = input_weight.loc[date]    # actual weight equals input weight
                nav.loc[date] = 1    # Normalize NAV to 1
                shares.loc[date] = actual_weight.loc[date] / data.loc[date]
                fee.loc[date] = self.calculate_fee(sb=pd.Series([0] * len(data.columns), index=data.columns),
                                                   sa=shares.loc[date], f=fee_rate, pa=data.loc[date])
                turnover_ratio.loc[date] = 0  # Initial transaction weight not considered part of turnover

            else:   # For the remaining dates...

                if date not in input_weight.index:    # If it's not a rebalancing date...
                    shares.loc[date] = shares.loc[data.index[idx - 1]]
                    # Shares unchanged, i.e., NOT rebalanced
                    fee.loc[date] = 0   # No fee incurred, as no rebalancing occurred
                    turnover_ratio.loc[date] = 0    # Turnover is 0, as no rebalancing occurred
                    nav.loc[date] = nav.loc[data.index[idx - 1]] + sum(shares.loc[date] *
                                                                       (data.loc[date] - data.loc[data.index[idx - 1]]))
                    # NAV increase comes from price increment sum times shares held over assets
                    actual_weight.loc[date] = shares.loc[date] * data.loc[date] / nav.loc[date]
                    # Actual weight = asset value over NAV

                else:   # If it's a rebalancing date...
                    shares_before = shares.loc[data.index[idx - 1]]
                    # Shares before rebalancing equals previous date shares
                    nav_before = nav.loc[data.index[idx - 1]] + sum(shares_before * (data.loc[date] -
                                                                                     data.loc[data.index[idx - 1]]))
                    try:
                        nav_after = newton(nav_equation, x0=nav_before, args=(shares_before, input_weight.loc[date],
                                                                          data.loc[date], fee_rate, nav_before),
                                       x1=nav_before*max(self.high_risk_fee_rate, self.low_risk_fee_rate))
                    except RuntimeError:
                        raise ValueError("Cannot generate NAV series due to unexpected value in data or weight.")
                    fee.loc[date] = nav_before - nav_after
                    nav.loc[date] = nav_after
                    actual_weight.loc[date] = input_weight.loc[date]
                    shares.loc[date] = actual_weight.loc[date] * nav_after / data.loc[date]
                    turnover_ratio.loc[date] = abs(input_weight.loc[date] -
                                                   actual_weight.loc[actual_weight.index[idx - 1]])

        # --------------------------------------------------------------------------------------------------------------
        # Step 4: Organize results
        portfolio_stats['Portfolio NAV'] = nav
        portfolio_stats['Transaction Fees'] = fee
        self.backtest_results['Backtest Summary'] = None
        self.backtest_results['NAV and Transaction Fees'] = portfolio_stats
        self.backtest_results['Normalized Asset Prices'] = data
        self.backtest_results['Shares Held (Normalized Prices)'] = shares
        self.backtest_results['Asset Weights'] = actual_weight
        self.backtest_results['Target Weights'] = input_weight
        self.backtest_results['Turnover Ratio'] = turnover_ratio
    
    def backtest(self):
        nav_backtest = Single_Asset(ann=self.ann, rf=self.rf,
                                    data=pd.DataFrame(self.backtest_results['NAV and Transaction Fees']['Portfolio NAV']))
        nav_backtest.backtest('Portfolio NAV')
        self.backtest_results['Backtest Summary'] = nav_backtest.backtest_results['Portfolio NAV']

        # --------------------------------------------------------------------------------------------------------------
        # Calculate turnover ratio
        years = list(set(self.backtest_results['Turnover Ratio'].index.year))
        years.sort()
        self.backtest_results['Backtest Summary']['Portfolio Turnover'] = 0
        for asset in self.backtest_results['Turnover Ratio'].columns:
            self.backtest_results['Backtest Summary'].loc['Overall Performance', asset + ' Turnover'] =\
                sum(self.backtest_results['Turnover Ratio'][asset])
            self.backtest_results['Backtest Summary'].loc['Overall Performance', 'Portfolio Turnover'] +=\
                self.backtest_results['Backtest Summary'].loc['Overall Performance', asset + ' Turnover']
            for idx, year in enumerate(years):
                if idx == 0:
                    # If the first year only involves one data point, it merely serves as the opening price of the next
                    # year's series and is included in next year's turnover
                    if len(self.backtest_results['Turnover Ratio'].loc[self.backtest_results['Turnover Ratio'].index.year == year,
                                                           asset]) == 1:
                        self.backtest_results['Backtest Summary'].loc[years[1], asset + ' Turnover'] = \
                            sum(self.backtest_results['Turnover Ratio'].loc[self.backtest_results['Turnover Ratio'].index.year
                                                                   == years[1], asset]) + \
                            sum(self.backtest_results['Turnover Ratio'].loc[self.backtest_results['Turnover Ratio'].index.year
                                                                   == years[0], asset])
                        continue
                        # "Continue" here is indispensable, or lines below will trigger a KeyError
                    else:
                        self.backtest_results['Backtest Summary'].loc[year, asset + ' Turnover'] = \
                            sum(self.backtest_results['Turnover Ratio'].loc[self.backtest_results['Turnover Ratio'].index.year
                                                                   == year, asset])
                elif idx == 1:
                    # If the first year only involves one data point, next year's turnover is already calculated
                    # in the above section
                    if len(self.backtest_results['Turnover Ratio'].loc[self.backtest_results['Turnover Ratio'].index.year
                                                              == years[0], asset]) > 1:
                        self.backtest_results['Backtest Summary'].loc[year, asset + ' Turnover'] = \
                            sum(self.backtest_results['Turnover Ratio'].loc[self.backtest_results['Turnover Ratio'].index.year
                                                                   == year, asset])
                else:
                    self.backtest_results['Backtest Summary'].loc[year, asset + ' Turnover'] = \
                        sum(self.backtest_results['Turnover Ratio'].loc[self.backtest_results['Turnover Ratio'].index.year
                                                               == year, asset])

                self.backtest_results['Backtest Summary'].loc[year, 'Portfolio Turnover'] += \
                    self.backtest_results['Backtest Summary'].loc[year, asset + ' Turnover']

    def output(self, output_path: str):
        """
        Save results as an Excel file
        :param str output_path: desired file path of the Excel file containing backtest results
        """
        writer = pd.ExcelWriter(path=output_path)
        for key in self.backtest_results.keys():
            temp = self.backtest_results[key]
            if key != 'Backtest Summary':
                temp.index = temp.index.date
            temp.to_excel(writer, sheet_name=key)
        writer.save()

if __name__ == "__main__":
    ann = 250
    rf = 0
    input_path = r'..\Test\05_Leverage_and_Short_Selling\data.xlsx'
    output_path = '..\Test\05_Leverage_and_Short_Selling\Portfolio_Backtest_Results.xlsx'
    high_risk_name_list = ['CSI 800 Index']  # Input high-risk assets according to Excel column names
    high_risk_fee_rate = 0  # Transaction fee
    low_risk_name_list = ['ChinaBond Composite Index', 'Non-standard']  # Input low-risk assets according to Excel column names
    low_risk_fee_rate = 0

    pb = Portfolio(ann=ann, rf=rf)
    pb.load_sheets_from_file(input_path=input_path, data_sheet_name='Data', weight_sheet_name='Weights')
    pb.load_fee_rates(high_risk_name_list=high_risk_name_list, high_risk_fee_rate=high_risk_fee_rate,
                      low_risk_name_list=low_risk_name_list, low_risk_fee_rate=low_risk_fee_rate)
    start_date = None
    end_date = None
    pb.slice(start_date, end_date)
    pb.generate_nav()
    pb.backtest()
    pb.output(output_path=output_path)
