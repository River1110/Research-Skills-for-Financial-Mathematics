# common library
import pandas as pd
import numpy as np
import time
import gym
import os

# RL models from stable-baselines
from stable_baselines import GAIL, SAC
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from EnvMultipleStock_train_copy import StockEnvTrain
from EnvMultipleStock_validation_copy import StockEnvValidation
from EnvMultipleStock_trade_1 import StockEnvTrade
from stable_baselines import DQN

def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_SAC(env_train, model_name, timesteps=25000):
    n_actions = env_train.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    start = time.time()
    model = TD3("MlpPolicy", env_train, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

# def train_SAC(env_train, model_name, timesteps=50000):
#     start = time.time()
#     model = ACER('MlpPolicy', env_train, verbose=0)
#     model.learn(total_timesteps=timesteps)
#     end = time.time()
#     model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
#     print('Training time (A2C): ', (end - start) / 60, ' minutes')
#     return model

def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

# def train_GAIL(env_train, model_name, timesteps=1000):
#     """GAIL Model"""
#
#     start = time.time()
#     # generate expert trajectories
#     model = SAC('MLpPolicy', env_train, verbose=1)
#     generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)
#
#     # Load dataset
#     dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
#     model = GAIL('MLpPolicy', env_train, dataset, verbose=1)
#
#     model.learn(total_timesteps=1000)
#     end = time.time()
#
#     model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
#     print('Training time (PPO): ', (end - start) / 60, ' minutes')
#     return model


def DRL_prediction(data,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_time,
                   turbulence_threshold,
                   initial):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(data, start=unique_trade_date[iter_num - rebalance_time], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    data_last_state = pd.DataFrame({'last_state': last_state})
    data_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    data_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    data_total_value.columns = ['account_value_train']
    data_total_value['daily_return'] = data_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * data_total_value['daily_return'].mean() / \
             data_total_value['daily_return'].std()
    return sharpe


def run_ensemble_strategy(data, unique_trade_date, rebalance_time, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    # validation_window is the number of days to validation the model and select for trading（三个月中的交易日）
    # rebalance_time is the number of days to retrain the model （每三个月重新训练模型）
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []
    sac_sharpe_list = []
    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []
    ppo_ensemble = []
    a2c_ensemble = []
    ddpg_ensemble = []
    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    # 用分位数根据in-sample data计算turbulence threshold
    insample_turbulence = data[(data.datadate<20170701) & (data.datadate>=20120100)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90) 

    start = time.time()
    #
    for i in range(rebalance_time + validation_window, len(unique_trade_date), rebalance_time):
        #每隔三个月训练一次，所以步长为rebalance_time
        print("============================================")
        ## initial state is empty
        if i - rebalance_time - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = data.index[data["datadate"] == unique_trade_date[i - rebalance_time - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*10 + 1 #30支股票

        historical_turbulence = data.iloc[start_date_index:(end_date_index + 1), :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(data, start=20120000, end=unique_trade_date[i - rebalance_time - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(data, start=unique_trade_date[i - rebalance_time - validation_window],
                                end=unique_trade_date[i - rebalance_time])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: ", 20120100, "to ",
              unique_trade_date[i - rebalance_time - validation_window])
        # print("training: ",len(data_split(data, start=20090000, end=test.datadate.unique()[i-rebalance_time]) ))
        # print("==============Model Training===========")
        print("======SAC Training========")
        model_SAC = train_SAC(env_train, model_name="SAC_dow_{}".format(i), timesteps=30000)
        print("======SAC Validation from: ", unique_trade_date[i - rebalance_time - validation_window], "to ",
              unique_trade_date[i - rebalance_time])
        DRL_validation(model=model_SAC, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_SAC = get_validation_sharpe(i)
        print("SAC Sharpe Ratio: ", sharpe_SAC)

        print("======A2C Training========")
        model_a2c = train_A2C(env_train, model_name="A2C_dow_{}".format(i), timesteps=30000)
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_time - validation_window], "to ",
              unique_trade_date[i - rebalance_time])
        DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print("A2C Sharpe Ratio: ", sharpe_a2c)

        print("======PPO Training========")
        model_ppo = train_PPO(env_train, model_name="PPO_dow_{}".format(i), timesteps=100000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_time - validation_window], "to ",
              unique_trade_date[i - rebalance_time])
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        print("======DDPG Training========")
        model_ddpg = train_DDPG(env_train, model_name="DDPG_dow_{}".format(i), timesteps=10000)
        print("======DDPG Validation from: ", unique_trade_date[i - rebalance_time - validation_window], "to ",
              unique_trade_date[i - rebalance_time])
        DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(i)



        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)
        sac_sharpe_list.append(sharpe_SAC)

        # Model Selection based on sharpe ratio
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg) & (sharpe_ppo >= sharpe_SAC):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg) & (sharpe_a2c >= sharpe_SAC):
            model_ensemble = model_a2c
            model_use.append('A2C')
        elif (sharpe_SAC > sharpe_ppo) & (sharpe_SAC > sharpe_ddpg) & (sharpe_SAC >= sharpe_ddpg):
            model_ensemble = model_SAC
            model_use.append('DDPG')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_time], "to ", unique_trade_date[i])
        #print("Used Model: ", model_ensemble)

        last_state_ensemble = DRL_prediction(data=data, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_time=rebalance_time,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        print("Used Model: ppo")
        ppo_ensemble = DRL_prediction(data=data, model=model_ppo, name="ppo",
                                             last_state=ppo_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_time=rebalance_time,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        print("Used Model: a2c")
        a2c_ensemble = DRL_prediction(data=data, model=model_a2c, name="a2c",
                                             last_state=a2c_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_time=rebalance_time,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        print("Used Model: ddpg")
        ddpg_ensemble = DRL_prediction(data=data, model=model_ddpg, name="ddpg",
                                             last_state=ddpg_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_time=rebalance_time,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        print("Used Model: sac")
        sac_ensemble = DRL_prediction(data=data, model=model_SAC, name="sac",
                                             last_state=sac_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_time=rebalance_time,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        # print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    ppo_sharpe = pd.DataFrame(data=ppo_sharpe_list)
    a2c_sharpe = pd.DataFrame(data=a2c_sharpe_list)
    ddpg_sharpe = pd.DataFrame(data=ddpg_sharpe_list)
    sac_sharpe = pd.DataFrame(data=sac_sharpe_list)
    ppo_sharpe.to_csv("sharpe_data/ppo_sharpe_list.csv")
    a2c_sharpe.to_csv("sharpe_data/a2c_sharpe_list.csv")
    ddpg_sharpe.to_csv("sharpe_data/ddpg_sharpe_list.csv")
    sac_sharpe.to_csv("sharpe_data/sac_sharpe_list.csv")
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
