class mr:
    def __init__(self, db, SMA=None, series=False, threshold=None):
        self.db = db
        self.series = series
        self.SMA = SMA
        self.threshold=threshold

    def MR_strategy(self, plot=False, data_testing_mode=False):
        import pandas as pd
        import numpy as np
        import os
        import matplotlib.pyplot as plt

        if plot:
            # create a dir for plots
            output_dir = "plots"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving plots in directory: {os.path.abspath(output_dir)}")

        # Check for missing values in the dataset
        if self.db.isna().any().any():
            print("Warning: The database has NAs. Results may vary due to missing data.")

        # Determine assets based on series flag
        assets = [self.db.name] if self.series else self.db.columns

        # Define dataset to use in the function
        d = self.db.copy(deep=True)
        if self.series:
            d = pd.DataFrame(d, columns=assets)

        # Compute returns
        for asset in assets:
            d[f'{asset}_Returns'] = np.log(d[asset] / d[asset].shift(1))

        for s in self.SMA:
            print("SUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA, ", s)
            for asset in assets:
                d[f'{asset}_SMA']=d[asset].rolling(window=s).mean()
                d[f'{asset}_distance']=d[asset]-d[f'{asset}_SMA']
                d[f'{asset}_position']=np.where(d[f'{asset}_distance']>self.threshold, -1, np.nan)
                d[f'{asset}_position']=np.where(d[f'{asset}_distance']<-self.threshold, 1, d[f'{asset}_position'])
                d[f'{asset}_position']=np.where(d[f'{asset}_distance']*d[f'{asset}_distance'].shift(1)<0, 0, d[f'{asset}_position'])
                d[f'{asset}_position']=d[f'{asset}_position'].ffill().fillna(0)

                d[f'{asset}_strategy']=d[f'{asset}_position'].shift(1)*d[f'{asset}_Returns']

                if plot:
                    plt.ioff()
                    d[f'{asset}_distance'].dropna().plot(figsize=(10, 6), legend=True)
                    plt.axhline(self.threshold, color='r')
                    plt.axhline(-self.threshold, color='r')
                    plt.axhline(0, color='r')
                    filename = f"{asset}_mean.png"
                    plt.savefig(os.path.join(output_dir, filename))
                    plt.close('all')

                    plt.ioff()
                    d[[f'{asset}_Returns', f'{asset}_strategy']].dropna().cumsum(
                    ).apply(np.exp).plot(figsize=(10, 6))
                    filename = f"{asset}_mean_returns.png"
                    plt.savefig(os.path.join(output_dir, filename))
                    plt.close('all')

                # cumulative results
                # only ran when in testing mode and not training
                # Check if `full_data_strategy` is initialized
                if 'full_data_strategy' not in locals():
                    full_data_strategy = pd.DataFrame()
                if data_testing_mode:
                    data_to_append = d[[f'{asset}_strategy']].cumsum().apply(np.exp)
                    full_data_strategy = pd.concat(
                        [full_data_strategy, data_to_append],
                        axis=1
                    )

                # aggregated results
                if 'results' not in locals():
                    results = pd.DataFrame()

                results = results.append(pd.DataFrame({
                    'SMA':s,
                    'ASSET':asset,
                    'Active_returns': np.exp(d[f'{asset}_strategy'].sum()),
                    'Passive_returns': np.exp(d[f'{asset}_Returns'].sum()),
                    'delta': np.exp(d[f'{asset}_strategy'].sum()) - np.exp(d[f'{asset}_Returns'].sum())
                }, index=[0]), ignore_index=True)

                # best sma
                idx = results.groupby('ASSET')['delta'].apply(lambda x: x.idxmax())
                best_MA = results.iloc[idx]
                sma = best_MA['SMA'].median()

        return {'results': results, 'full_data': full_data_strategy, 'best_MO': best_MA, 'sma_median': sma}

    def mr_backtesting(self, percent_training=0.7):

        # Sequentially split the dataset
        split_index = int(len(self.db) * percent_training)  # 70% training, 30% testing
        db_train = self.db.iloc[:split_index]  # Training set
        db_test = self.db.iloc[split_index:]  # Testing set

        # Step 1: Optimize SMA parameters on the training set
        self.db = db_train

        training_results = self.MR_strategy()

        sma_median = training_results['sma_median']
        print("\nTraining Results:")
        print(training_results['results'])

        print("Optimal Parameters from Training:")
        print(f"MO Median: {sma_median}")

        # Step 2: Backtest on the testing set with optimal parameters
        self.sma = [int(sma_median)]
        self.db = db_test
        testing_results = self.MR_strategy(plot=True, data_testing_mode=True)

        print("\nTesting Results:")
        print(testing_results['results'])

        return {'testing_results':testing_results, 'training_results':training_results}
