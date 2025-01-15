class SMA:
    def __init__(self, db, SMA1, SMA2, series):
        self.db = db
        self.sma1 = SMA1
        self.sma2 = SMA2
        self.series = series
        self.testing_results = []
        self.weights = []

    def MA_strategy_short(self, output_print=False, plot=True, full_data_out=False):
        import pandas as pd
        import numpy as np
        from itertools import product
        import matplotlib.pyplot as plt
        import os
        import matplotlib
        matplotlib.use('Agg')

        if plot:
            # create a dir for plots
            output_dir = "plots"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving plots in directory: {os.path.abspath(output_dir)}")

        results = pd.DataFrame()

        # Check for missing values in the dataset
        if self.db.isna().any().any():
            print("Warning: The database has NAs. Results may vary due to missing data.")

        # Determine assets based on series flag
        assets = [self.db.name] if self.series else self.db.columns

        # Brute force: Calculate moving averages and strategy results
        for SMA1, SMA2 in product(self.sma1, self.sma2):
            d = self.db.copy(deep=True)
            if self.series:
                d = pd.DataFrame(d, columns=assets)

            for col in assets:
                d[f'{col}_MA1'] = d[col].rolling(window=SMA1).mean()
                d[f'{col}_MA2'] = d[col].rolling(window=SMA2).mean()
                d[f'{col}_Position'] = np.where(d[f'{col}_MA1'] > d[f'{col}_MA2'], 1, -1)

            d.dropna(inplace=True)

            for asset in assets:
                d[f'{asset}_Returns'] = np.log(d[asset] / d[asset].shift(1))
                d[f'{asset}_Strategy_yes_short'] = d[f'{asset}_Position'].shift(1) * d[f'{asset}_Returns']

                if plot:
                    plt.ioff()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.set_title(f"Prices : SMA1: {SMA1}, SMA2: {SMA2}")
                    ax.legend(loc="best")
                    filename = f"Prices_{asset}_SMA1-{SMA1}_SMA2-{SMA2}.png"
                    d[[asset, f'{asset}_MA1', f'{asset}_MA2', f'{asset}_Position']].plot(
                        secondary_y=f'{asset}_Position')
                    plt.savefig(os.path.join(output_dir, filename))
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.set_title(f"Returns : SMA1: {SMA1}, SMA2: {SMA2}")
                    ax.legend(loc="best")
                    filename = f"Returns_{asset}_SMA1-{SMA1}_SMA2-{SMA2}.png"
                    d[[f'{asset}_Returns', f'{asset}_Strategy_yes_short']].cumsum().apply(np.exp).plot()
                    #np.exp(temp).cumsum().plot()
                    plt.savefig(os.path.join(output_dir, filename))
                    plt.close(fig)
                plt.close('all')
                if output_print:
                    print(f"******** {SMA1} *** {SMA2} ********")
                    print(f"Asset: {asset}")
                    #print("Returns:", np.exp(d[[f'{asset}_Returns', f'{asset}_Strategy_no_short', f'{asset}_Strategy_yes_short']].sum()))
                    print("Returns:", np.exp(d[[f'{asset}_Returns', f'{asset}_Strategy_yes_short']].sum()))
                    print("Volatility:", np.exp(d[[f'{asset}_Returns', f'{asset}_Strategy_yes_short']].std() * 252**0.5))

                # cumulative results
                # Ensure d[[...]] results in a DataFrame or Series as expected
                data_to_append = d[[f'{asset}_Strategy_yes_short']].cumsum().apply(np.exp)

                # Check if `full_data_strategy` is initialized
                if 'full_data_strategy' not in locals():
                    full_data_strategy = pd.DataFrame()

                if full_data_out:
                    full_data_strategy = pd.concat(
                        [full_data_strategy, data_to_append],
                        axis=1
                    )

                # aggregated results
                results = results.append(pd.DataFrame({
                    'SMA1': SMA1, 'SMA2': SMA2, 'ASSET': asset,
                    'STRATEGY_YES_SHORT': np.exp(d[f'{asset}_Strategy_yes_short'].sum()),
                    'Returns': np.exp(d[f'{asset}_Returns'].sum()),
                    'V_YES_SHORT': np.exp(d[f'{asset}_Strategy_yes_short'].std() * 252**0.5),
                    'delta': np.exp(d[f'{asset}_Strategy_yes_short'].sum()) - np.exp(d[f'{asset}_Returns'].sum()),
                }, index=[0]), ignore_index=True)

        # Identify best MA parameters
        idx = results.groupby('ASSET')['delta'].apply(lambda x: x.idxmax())
        best_MA = results.iloc[idx]
        ma1 = best_MA['SMA1'].median()
        ma2 = best_MA['SMA2'].median()

        full_data= [d] if full_data_out else []

        return {'results': results, 'best_MA': best_MA, 'MA1_median': ma1, 'MA2_median': ma2, 'full_data':full_data_strategy}

    def MA_strategy_long(self, output_print=False, plot=True, full_data_out=False):
        import pandas as pd
        import numpy as np
        from itertools import product
        import matplotlib.pyplot as plt
        import os
        import matplotlib
        matplotlib.use('Agg')

        if plot:
            # create a dir for plots
            output_dir = "plots"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving plots in directory: {os.path.abspath(output_dir)}")

        results = pd.DataFrame()

        # Check for missing values in the dataset
        if self.db.isna().any().any():
            print("Warning: The database has NAs. Results may vary due to missing data.")

        # Determine assets based on series flag
        assets = [self.db.name] if self.series else self.db.columns

        # Brute force: Calculate moving averages and strategy results
        for SMA1, SMA2 in product(self.sma1, self.sma2):
            d = self.db.copy(deep=True)
            if self.series:
                d = pd.DataFrame(d, columns=assets)

            for col in assets:
                d[f'{col}_MA1'] = d[col].rolling(window=SMA1).mean()
                d[f'{col}_MA2'] = d[col].rolling(window=SMA2).mean()
                d[f'{col}_Position'] = np.where(d[f'{col}_MA1'] > d[f'{col}_MA2'], 1, 0)  # No short

            d.dropna(inplace=True)

            for asset in assets:
                d[f'{asset}_Returns'] = np.log(d[asset] / d[asset].shift(1))
                d[f'{asset}_Strategy_no_short'] = d[f'{asset}_Position'].shift(1) * d[f'{asset}_Returns']

                if plot:
                    plt.ioff()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.set_title(f"Prices : SMA1: {SMA1}, SMA2: {SMA2}")
                    ax.legend(loc="best")
                    filename = f"Prices_{asset}_SMA1-{SMA1}_SMA2-{SMA2}.png"
                    d[[asset, f'{asset}_MA1', f'{asset}_MA2', f'{asset}_Position']].plot(
                        secondary_y=f'{asset}_Position')
                    plt.savefig(os.path.join(output_dir, filename))
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.set_title(f"Returns : SMA1: {SMA1}, SMA2: {SMA2}")
                    ax.legend(loc="best")
                    filename = f"Returns_{asset}_SMA1-{SMA1}_SMA2-{SMA2}.png"
                    d[[f'{asset}_Returns', f'{asset}_Strategy_no_short']].cumsum().apply(np.exp).plot()
                    # np.exp(temp).cumsum().plot()
                    plt.savefig(os.path.join(output_dir, filename))
                    plt.close(fig)
                plt.close('all')

                if output_print:
                    print(f"******** {SMA1} *** {SMA2} ********")
                    print(f"Asset: {asset}")
                    print("Returns:", np.exp(d[[f'{asset}_Returns', f'{asset}_Strategy_no_short']].sum()))
                    print("Volatility:", np.exp(d[[f'{asset}_Returns', f'{asset}_Strategy_no_short']].std() * 252**0.5))

                # cumulative results
                # Ensure d[[...]] results in a DataFrame or Series as expected
                data_to_append = d[[f'{asset}_Strategy_no_short']].cumsum().apply(np.exp)

                # Check if `full_data_strategy` is initialized
                if 'full_data_strategy' not in locals():
                    full_data_strategy = pd.DataFrame()

                if full_data_out:
                    full_data_strategy = pd.concat(
                        [full_data_strategy, data_to_append],
                        axis=1
                    )

                # aggregated results
                results = results.append(pd.DataFrame({
                    'SMA1': SMA1, 'SMA2': SMA2, 'ASSET': asset,
                    'STRATEGY_NO_SHORT': np.exp(d[f'{asset}_Strategy_no_short'].sum()),
                    'Returns': np.exp(d[f'{asset}_Returns'].sum()),
                    'V_NO_SHORT': np.exp(d[f'{asset}_Strategy_no_short'].std() * 252**0.5),
                    'delta': np.exp(d[f'{asset}_Strategy_no_short'].sum()) - np.exp(d[f'{asset}_Returns'].sum()),
                    #'strategy': "LONG" if np.exp(d[f'{asset}_Strategy_no_short'].sum()) > np.exp(d[f'{asset}_Strategy_yes_short'].sum()) else "SHORT"
                }, index=[0]), ignore_index=True)

        # Identify best MA parameters
        idx = results.groupby('ASSET')['delta'].apply(lambda x: x.idxmax())
        best_MA = results.iloc[idx]
        ma1 = best_MA['SMA1'].median()
        ma2 = best_MA['SMA2'].median()

        full_data = [d] if full_data_out else []

        return {'results': results, 'best_MA': best_MA, 'MA1_median': ma1, 'MA2_median': ma2,
                'full_data': full_data_strategy}

    def ma_backtesting(self, short_allowed=True, plot=True, percent_training=0.7):

        # Sequentially split the dataset
        split_index = int(len(self.db) * percent_training)  # 70% training, 30% testing
        db_train = self.db.iloc[:split_index]  # Training set
        db_test = self.db.iloc[split_index:]  # Testing set

        # Step 1: Optimize SMA parameters on the training set
        self.db = db_train.copy(deep=True)
        if short_allowed:
            training_results = self.MA_strategy_short(output_print=False, plot=False)
        else:
            training_results = self.MA_strategy_long(output_print=False, plot=False)

        ma1_median = training_results['MA1_median']
        ma2_median = training_results['MA2_median']

        print("\nTraining Results:")
        print(training_results['results'])

        print("Optimal Parameters from Training:")
        print(f"MA1 Median: {ma1_median}")
        print(f"MA2 Median: {ma2_median}")

        # Step 2: Backtest on the testing set with optimal parameters
        self.sma1 = [int(ma1_median)]
        self.sma2 = [int(ma2_median)]
        self.db = db_test.copy(deep=True)
        if short_allowed:
            testing_results = self.MA_strategy_short(output_print=True, plot=plot, full_data_out=True)
        else:
            testing_results = self.MA_strategy_long(output_print=True, plot=plot, full_data_out=True)

        self.testing_results=testing_results

        print("\nTesting Results:")
        print(testing_results['results'])

        return {'testing_results':testing_results, 'training_results':training_results}

    def portfolio_simulation(self, weights=None, dumb_strategy=None, benchmark=None, plot_name=None):
        import numpy as np
        import import_data
        import matplotlib.pyplot as plt

        if not hasattr(self, 'testing_results'):
            raise AttributeError("No testing results available. Run 'ma_backtesting' first.")

        # get daily results from strategy
        d=self.testing_results['full_data'].copy(deep=True)
        d['portfolio_returns']=np.average(d.values, axis=1, weights=weights)
        if dumb_strategy == "even":
            d['dumb_strategy'] = np.average(d.values, axis=1, weights=np.array(len(d.columns) * [1. / len(d.columns),]))

        benchmark_prices = import_data.datosYahoo(asset_list=[benchmark], start=d.index.min())
        benchmark_rets = np.log(benchmark_prices / benchmark_prices.shift(1))
        d['benchmark'] = benchmark_rets[benchmark].cumsum().apply(np.exp)

        d[['portfolio_returns', 'benchmark', 'dumb_strategy']].plot()
        plt.savefig(f'{plot_name}_benchmark_comparison.png')

        return d
