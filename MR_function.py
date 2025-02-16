class MO:
    def __init__(self, db, momentum, series):
        self.db = db
        self.series = series
        self.momentum=momentum


    def MO_strategy(self, plot=False, data_testing_mode=False):
        import pandas as pd
        import numpy as np
        import os

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
            print(d.head())

        # brute force
        for m in self.momentum:

            # loop over assets and create signals + returns
            for asset in assets:
                d['momentum'] = m
                d[f'strategy_{asset}'] = np.sign(d[f'{asset}_Returns'].rolling(m).mean())
                d[f'strategy_returns_{asset}'] = d[f'strategy_{asset}'].shift(1) * d[f'{asset}_Returns']

                # cumulative results
                # only ran when in testing mode and not training

                # Check if `full_data_strategy` is initialized
                if 'full_data_strategy' not in locals():
                    full_data_strategy = pd.DataFrame()

                if data_testing_mode:
                    data_to_append = d[[f'strategy_returns_{asset}']].cumsum().apply(np.exp)
                    full_data_strategy = pd.concat(
                        [full_data_strategy, data_to_append],
                        axis=1
                    )

                # aggregated results
                if 'results' not in locals():
                    results = pd.DataFrame()

                results = results.append(pd.DataFrame({
                    'momentum': m, 'ASSET': asset,
                    'Active_returns': np.exp(d[f'strategy_returns_{asset}'].sum()),
                    'Passive_returns': np.exp(d[f'{asset}_Returns'].sum()),
                    'delta': np.exp(d[f'strategy_returns_{asset}'].sum()) - np.exp(d[f'{asset}_Returns'].sum()),
                }, index=[0]), ignore_index=True)

                # best momentum
                idx = results.groupby('ASSET')['delta'].apply(lambda x: x.idxmax())
                best_MO = results.iloc[idx]
                mo = best_MO['momentum'].median()

        return {'results':results, 'full_data':full_data_strategy, 'best_MO':best_MO, 'mo_median':mo}

    def mo_backtesting(self, percent_training=0.7):

        # Sequentially split the dataset
        split_index = int(len(self.db) * percent_training)  # 70% training, 30% testing
        db_train = self.db.iloc[:split_index]  # Training set
        db_test = self.db.iloc[split_index:]  # Testing set

        # Step 1: Optimize SMA parameters on the training set
        self.db = db_train

        training_results = self.MO_strategy()

        mo_median = training_results['mo_median']
        print("\nTraining Results:")
        print(training_results['results'])

        print("Optimal Parameters from Training:")
        print(f"MO Median: {mo_median}")

        # Step 2: Backtest on the testing set with optimal parameters
        self.momentum = [int(mo_median)]
        self.db = db_test
        testing_results = self.MO_strategy(data_testing_mode=True)

        print("\nTesting Results:")
        print(testing_results['results'])

        return {'testing_results':testing_results, 'training_results':training_results}

