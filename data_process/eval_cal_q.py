import pandas as pd
import numpy as np
from numpy import linalg as LA

const_q = 0.5
sampling_period = 0.5
T = 200
W = const_q * np.concatenate((
    np.concatenate((sampling_period ** 3 / 3 * np.eye(2),
                    sampling_period ** 2 / 2 * np.eye(2)), axis=1),
    np.concatenate((sampling_period ** 2 / 2 * np.eye(2),
                    sampling_period * np.eye(2)), axis=1)))
J = []
Phi = []
J_max = -T * np.log(LA.det(W))
for i in range(10):
    # df = pd.read_csv('../../data_record/'+'model_q_'+str(const_q)+'_'+str(i+1)+'.csv',
    #                  names=['prior_data', 'posterior_data', 'observed'])  # 使用制表符分隔
    df = pd.read_csv('../../data_record/'+'greedy_500q_'+str(const_q)+'_'+str(i+1)+'.csv',
                     names=['prior_data', 'posterior_data', 'observed'])  # 使用制表符分隔
    prior_data = df['prior_data'].to_numpy()
    posterior_data = df['posterior_data'].to_numpy()
    observed = df['observed'].to_numpy()

    J_min = np.sum(prior_data)
    J_bar = (np.sum(posterior_data) - J_min)/(J_max - J_min)
    # J_min = np.sum(posterior_data)
    # J_bar = (np.sum(prior_data) - J_min) / (J_max - J_min)
    J.append(J_bar)

    true_count = np.sum(observed)
    Phi.append(true_count/T)
print(np.mean(J), '\n', np.std(J))
print(np.mean(Phi))