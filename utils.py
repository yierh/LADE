import numpy as np
from globalVar import *
import matplotlib.pyplot as pl
import torch
import datetime


class MyExcept(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


def inspect_hyper(func17, reward_func, lambda_value):
    if func17 != 0:
        raise MyExcept('This is 13 test, 17_flag should be 0!')
    elif reward_func == 3 and lambda_value == 0:
        raise MyExcept('r1+r2 need a weight which is not None!')
    elif reward_func != 3 and lambda_value != 0:
        raise MyExcept('r1 and r2  do not need a reward func weight!')
    return 0


def get_func_index(i, train_flag, func17_flag):
    if func17_flag == 0:
        train_func = np.array([1, 3, 5, 6, 7, 9, 10,
                               11, 12, 13, 14, 17, 18, 19, 20,
                               31, 32, 33, 34, 35, 36])
        labels = ['f1', 'f3', 'f5', 'f6', 'f7', 'f9', 'f10',
                  'f11', 'f12', 'f13', 'f14', 'f17', 'f18', 'f19', 'f20',
                  'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
        test_func = np.array([2, 4, 8, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28])
    else:

        train_func = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                               11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                               21, 22, 23, 24, 25, 26, 27, 28,
                               31, 32, 33, 34, 35, 36])
        labels = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
                  'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',
                  'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28',
                  'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
        test_func = np.array([1, 3, 4, 5, 6, 7, 8, 9, 10,
                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                              21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    if train_flag == 1:
        func_index = train_func[i]
    else:
        func_index = test_func[i]
    # if train_flag == 1 and func17_flag == 1 and len(train_func) != PROBLEM_NUM:
    #     print('The number of training functions for CEC17 is not equal to what in settings!')
    return func_index, labels


def order_by_f(pop, fit):
    sorted_array = np.argsort(fit.flatten())  # [N,]，
    temp_pop = pop[sorted_array]  # [N,1]
    temp_fit = fit[sorted_array]  # temp_update_re = update_re[sorted_array]
    return temp_pop, temp_fit


def maxmin_norm(a):
    if np.max(a) != np.min(a):
        a = (a-np.min(a))/(np.max(a)-np.min(a))
    return a


def con2mat_F(f_vec):
    f_mat = np.zeros((POP_SIZE, POP_SIZE))
    for i in range(POP_SIZE):
        for j in range(POP_SIZE):
            f_mat[i, j] = f_vec[0, i]
    return f_mat


def generate_pop(population_size, input_dimension, x_min, x_max):
    pop = np.zeros((population_size, input_dimension)) # N*D
    for i in range(population_size):
        for j in range(input_dimension):
            pop[i, j] = x_min + np.random.uniform() * (x_max - x_min)
    return pop


def mulgenerate_pop(p, population_size, input_dimension, x_min, x_max):
    for i in range(p):
        if i == 0:
            pop = generate_pop(population_size, input_dimension, x_min, x_max)
        else:
            pop_c = generate_pop(population_size, input_dimension, x_min, x_max)
            pop = np.vstack((pop, pop_c))
    return pop.reshape(-1, population_size, input_dimension)


def plot_fit_2reward(p, x, fit, r1, r2, dbt_r1, dbt_r2, dbt_r3, filename, f_name):
    # func = get_func_index(p, train_flag=1, func17_flag=cec17_flag)
    pl.figure(p+1, figsize=(20, 10), dpi=100)
    pl.subplot(2, 3, 1)
    pl.plot(x, np.array(fit), color='green', label='function value')
    pl.legend()
    pl.title('Mean Best Fitness for all episodes at T of the Population - '+f_name)
    pl.xlabel('t')
    pl.ylabel('mean best fit')

    pl.subplot(2, 3, 2)
    pl.plot(x, np.array(r1), color='red', label='-mean r1')
    pl.legend()
    pl.title('Mean Relative Reward1 of the Population - '+f_name)
    pl.xlabel('t')
    pl.ylabel('R')

    pl.subplot(2, 3, 3)
    pl.plot(x, np.array(r2), color='orchid', label='-mean r2')
    pl.legend()
    pl.title('Mean Reward2 of the Population - ' + f_name)
    pl.xlabel('t')
    pl.ylabel('R')

    pl.subplot(2, 3, 4)
    pl.plot(x, np.array(dbt_r3), color='blue', label='-mean r3')
    pl.legend()
    pl.title('Mean Reward3 over the trajectories -' + f_name)
    pl.xlabel('t')
    pl.ylabel('R')

    pl.subplot(2, 3, 5)
    pl.plot(x, np.array(dbt_r1), color='darkred', label='-mean r1_dn')
    pl.legend()
    pl.title('Mean Relative Reward1 after DBT - ' + f_name)
    pl.xlabel('t')
    pl.ylabel('R')

    pl.subplot(2, 3, 6)
    pl.plot(x, np.array(dbt_r2), color='purple', label='-mean r2_dn')
    pl.legend()
    pl.title('Mean Reward2 after DBT - ' + f_name)
    pl.xlabel('t')
    pl.ylabel('R')

    pl.savefig(filename+'/'+f_name+'.png')


def plot_3sigmas_f_cr_w(x, sigma_f, sigma_cr, sigma_w, filename):
    pl.figure(figsize=(20, 10), dpi=100)
    pl.subplot(2, 2, 1)
    pl.plot(x, sigma_f, color='darkcyan', label='sigma_f')
    pl.legend()
    pl.title('Mean sigma of F for each problem in the training')
    pl.xlabel('t')
    pl.ylabel('Mean sigma_F')

    pl.subplot(2, 2, 2)
    pl.plot(x, sigma_cr, color='blue', label='sigma_cr')
    pl.legend()
    pl.title('Mean sigma of CR for each problem in the training')
    pl.xlabel('t')
    pl.ylabel('Mean sigma_CR')

    pl.subplot(2, 2, 3)
    pl.plot(x, sigma_w, color='crimson', label='sigma_w')
    pl.legend()
    pl.title('Mean sigma of W elements for each problem in the training')
    pl.xlabel('t')
    pl.ylabel('Mean sigma_w')
    pl.savefig(filename + '/3sigma.png')


def add_random(m_pop, pop, mu): # mutation pop = pop + F*W*pop + rand, m_pop = pop + F*W*pop
    mur_pop = np.zeros((pop.shape[0], pop.shape[1]))
    # mur_pop = np.zeros_like(pop)
    for i in range(pop.shape[0]):
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == i:
            r1 = np.random.randint(0, pop.shape[0])
        while r2 == i or r2 == r1:
            r2 = np.random.randint(0, pop.shape[0])
        mur_pop[i, :] = m_pop[i, :] + mu[0, i]*(pop[r1, :] - pop[r2, :])
    return mur_pop


def con2mat_norm_Pw_wo_loop_v3(w_samples):  # [N,P]->[N,N]
    w_mat = np.hstack((w_samples, np.zeros((w_samples.shape[0], POP_SIZE-w_samples.shape[1]))))  #[P+1, N]列补0
    w_sum = np.sum(w_mat, 1)
    for i in range(w_mat.shape[0]):
        if w_sum[i] == 0:
            w_sum[i] = 1.
        w_mat[i, :] /= w_sum[i]
        w_mat[i, i] += -1.
    return w_mat


def maxmin_norm_reward2(r):
    r = np.array(r).reshape(PROBLEM_NUM, -1)  # r[P,L]
    for i in range(r.shape[0]):
        r[i, :] = maxmin_norm(r[i, :])
    return r.flatten()


def r3_func(r1, r2):
    r1 = (np.hstack(r1)).reshape(PROBLEM_NUM*TRAJECTORY_NUM, -1)
    r2 = r2.reshape(-1, 1)
    r3 = r1+r2
    return r3.flatten()


def discounted_with_baseline_trunc_rewards(r, trunc0_flag, norm_flag):
    r = np.hstack(r)  # r: list[L*P*T]
    for ep in range(TRAJECTORY_NUM*PROBLEM_NUM):
        single_rs = r[ep*TRAJECTORY_LENGTH: ep*TRAJECTORY_LENGTH+TRAJECTORY_LENGTH]
        discounted_rs = np.zeros_like(single_rs)
        running_add = 0.
        for t in reversed(range(0, TRAJECTORY_LENGTH)):
            running_add = running_add * GAMMA + single_rs[t]
            discounted_rs[t] = running_add

        if ep == 0:
            all_disc_rs = discounted_rs
        else:
            all_disc_rs = np.hstack((all_disc_rs, discounted_rs))  # [P*L,T],[P*L*T]

    all_disc_rs = all_disc_rs.reshape(PROBLEM_NUM, TRAJECTORY_NUM, -1)  # [P,L,T]
    all_disc_norm_rs = np.zeros_like(all_disc_rs)
    if norm_flag == 1:  # mean normalization
        for i in range(PROBLEM_NUM):
            all_disc_norm_rs[i, :, :] = all_disc_rs[i, :, :] - np.mean(all_disc_rs[i, :, :], 0)
    if trunc0_flag == 1:
        all_disc_norm_rs = np.where(all_disc_norm_rs > 0.0, all_disc_norm_rs, 0.0)  # negtive -> 0
    return all_disc_norm_rs.flatten()


def baseline_trunc_expand_reward2(r, trunc0_flag, norm_flag):
    r = np.array(r).reshape(PROBLEM_NUM, -1)  # r[P*L]
    r2 = np.zeros_like(r)
    if norm_flag == 1:
        for i in range(r.shape[0]):
            r2[i, :] = r[i, :] - np.mean(r[i, :])
    if trunc0_flag == 1:  # relu
        r2 = np.where(r2 > 0.0, r2, 0.0)
    r2 = r2.flatten()
    rs = np.zeros((PROBLEM_NUM*TRAJECTORY_NUM, TRAJECTORY_LENGTH))
    for i in range(PROBLEM_NUM*TRAJECTORY_NUM):
        rs[i, :] = r2[i]
    return rs.flatten()



def discount_baseline_trunc_rewards(r1, r2, weight, norm_flag, trunc0_flag):
    for ep in range(TRAJECTORY_NUM*PROBLEM_NUM):  # r1∈[0,1]
        single_r1 = r1[ep*TRAJECTORY_LENGTH: ep*TRAJECTORY_LENGTH+TRAJECTORY_LENGTH]  # pick one reward of one trajectory in P*L
        discounted_r1 = np.zeros_like(single_r1)
        running_add = 0.
        for t in reversed(range(0, TRAJECTORY_LENGTH)):
            running_add = running_add * GAMMA + single_r1[t]
            discounted_r1[t] = running_add

        if ep == 0:
            all_disc_r1 = discounted_r1
        else:
            all_disc_r1 = np.hstack((all_disc_r1, discounted_r1)) #[P*L,T],[P*L*T]
    # if norm_r1_flag == 1:
    #     all_disc_norm_r1 = all_disc_r1 / TRAJECTORY_LENGTH
    rs1 = all_disc_r1.reshape(PROBLEM_NUM, TRAJECTORY_NUM, -1)  #

    r2 = r2.flatten()  # 这里经过max min归一化的r2∈[0,1]
    rs2 = np.zeros((PROBLEM_NUM * TRAJECTORY_NUM, TRAJECTORY_LENGTH))
    for i in range(PROBLEM_NUM * TRAJECTORY_NUM):
        rs2[i, :] = r2[i]
    rs2 = rs2.reshape(PROBLEM_NUM, TRAJECTORY_NUM, -1)

    r3 = rs1 + weight * rs2
    norm_r3 = np.zeros_like(r3)
    if norm_flag == 1:
        for i in range(PROBLEM_NUM):
            norm_r3[i, :, :] = r3[i, :, :] - np.mean(r3[i, :, :], 0)
    if trunc0_flag == 1:
        norm_r3 = np.where(norm_r3 > 0.0, norm_r3, 0.0)  # negative -> 0
    return norm_r3.flatten()
