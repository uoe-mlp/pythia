from __future__ import annotations
from typing import List, Union, Dict, Tuple
import pandas as pd
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.core.frame import DataFrame
import numpy as np
import os
from pandas_datareader import data as web

from pythia.journal import TradeOrder, TradeFill
from pythia.utils import ArgsParser

from .daily_historical_market import DailyHistoricalMarket


class LiveDailyHistoricalMarket(DailyHistoricalMarket):

    def __init__(self, X: np.ndarray, Y: np.ndarray, timestamps: List[pd.Timestamp], trading_cost: float, features: List[str], targets: List[str],
        download_timestamp: Timestamp, source: str, start_date: Timestamp, end_date: Timestamp, feature_keys: List[str], target_keys: List[str],
        instruments: List[str]):
        super(LiveDailyHistoricalMarket, self).__init__(X=X, Y=Y, timestamps=timestamps, trading_cost=trading_cost, 
            features_paths=[os.path.join('data', 'markets', source.lower(), '%s_%s_%s.csv' % (x.lower(), start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))) for x in features], 
            target_paths=[os.path.join('data', 'markets', source.lower(), '%s_%s_%s.csv' % (x.lower(), start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))) for x in targets],
            instruments=instruments)
        self.download_timestamp: Timestamp = download_timestamp
        self.assets: List[str] = sorted(list(set(features + targets)))
        self.features: List[str] = features
        self.targets: List[str] = targets
        self.source: str = source
        self.start_date: Timestamp = start_date
        self.end_date: Timestamp = end_date
        self.feature_keys: List[str] = feature_keys
        self.target_keys: List[str] = target_keys

    @staticmethod
    def initialise(params: Dict) -> LiveDailyHistoricalMarket:
        # Read the parameters from the dictionary
        features_raw: Union[List[str], str] = ArgsParser.get_or_error(params, 'features')
        features: List[str] = [features_raw] if isinstance(features_raw, str) else features_raw

        targets_raw: Union[List[str], str] = ArgsParser.get_or_error(params, 'targets')
        targets: List[str] = [targets_raw] if isinstance(targets_raw, str) else targets_raw

        source: str = ArgsParser.get_or_error(params, 'source')
        start_date: Timestamp = Timestamp(ArgsParser.get_or_error(params, 'start_date'))
        end_date: Timestamp = Timestamp(ArgsParser.get_or_error(params, 'end_date'))

        features_keys_raw: Union[List[str], str] = ArgsParser.get_or_default(params, 'feature_keys', 'Close')
        feature_keys: List[str] = [features_keys_raw] if isinstance(features_keys_raw, str) else features_keys_raw

        targets_keys_raw: Union[List[str], str] = ArgsParser.get_or_default(params, 'target_keys', 'Close')
        target_keys: List[str] = [targets_keys_raw] if isinstance(targets_keys_raw, str) else targets_keys_raw

        # Get timeeseries (filename is "ticker_yyyymmdd_yyyymmdd.csv", with start date first)
        t_df_arr: List[pd.DataFrame] = []
        f_df_arr: List[pd.DataFrame] = []
        for asset in sorted(list(set(features + targets))):

            pathname = os.path.join('data', 'markets', source.lower(), '%s_%s_%s.csv' % (asset.lower(), start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')))
            
            if os.path.isfile(pathname) == True:
                print('Loading from file: ', pathname)        
                df = pd.read_csv(pathname, index_col='Date')
            else:
                print('Downloading from Yahoo! - ', asset)
                df = web.DataReader(asset, data_source=source, start=start_date, end=end_date)
                df.to_csv(pathname)

            # Just before we put it in the dictionary, copy the index to a Date field and make a new index which enumerates
            # all the entries.
            df.insert(0, 'Date', df.index)
            df.index = np.arange(df.shape[0])
            df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d').apply(lambda x: Timestamp(x))
            df.rename({'Date':'date'}, axis=1, inplace=True)
            df.set_index('date', inplace=True)
            
            prev_feature_keys = [x[6:] for x in feature_keys if x[:6] == 'Prev. ']
            non_prev_feature_keys = [x for x in feature_keys if x[:6] != 'Prev. ']

            if asset in features:
                non_prev_df = df[non_prev_feature_keys]
                prev_df = df[prev_feature_keys]
                prev_df = pd.DataFrame(data=prev_df.values[:-1, :], index=prev_df.index[1:], columns=['Prev. %s' % (x) for x in prev_df.columns])
                f_df = pd.concat([non_prev_df, prev_df], axis=1)
                f_df.dropna(axis=0, inplace=True)
                f_df_arr.append(f_df)

            if asset in targets:
                tmp = df[target_keys]
                if tmp.shape[1] > 1:
                    tmp.columns = [x + asset for x in tmp.columns]
                else:
                    tmp.columns = [asset]
                t_df_arr.append(tmp)

        X, Y, dates, target_names = DailyHistoricalMarket.combine_datasets(f_df_arr, t_df_arr)

        trading_cost: float = ArgsParser.get_or_default(params, 'trading_cost', 1e-3)     

        return LiveDailyHistoricalMarket(X, Y, dates, trading_cost, features, targets, Timestamp.now(), source, start_date, end_date, feature_keys, target_keys, target_names)
