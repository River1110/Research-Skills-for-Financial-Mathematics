# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv
import warnings

warnings.filterwarnings('ignore')
# preprocessor
from preprocessors import *
# config
from config.config import *
# model
from models_copy import *
import os

def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "data-2.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
        data = data.sort_values(['datadate','tic']).reset_index(drop=True)
    # else:
    #     data = preprocess_data()
    #     data = add_turbulence(data)
    #     data.to_csv(preprocessed_path)
    print(data.head())
    print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20170701) & (data.datadate <= 20220101)].datadate.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window =  126
    validation_window = 126
    
    ## Ensemble Strategy
    run_ensemble_strategy(df=data,
                          unique_trade_date= unique_trade_date,
                          rebalance_window = rebalance_window,
                          validation_window = validation_window)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()