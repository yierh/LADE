import torch
import torch.nn as nn
from globalVar import *
import numpy as np
from utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PGAgent(nn.Module):
    def __init__(self, cellsize_f, cellsize_cr, cellsize_w):
        super(PGAgent, self).__init__()
        self.lstm_f = nn.LSTMCell(POP_SIZE+4, cellsize_f)  # (input size, hidden size, bias=True)
        self.linear_mu_f = nn.Linear(cellsize_f, NUM_F_MEAN)   # output F (mu) needed, here firstly try N means
        self.linear_sigma_f = nn.Linear(cellsize_f, 1)

        self.lstm_cr = nn.LSTMCell(POP_SIZE+4, cellsize_cr)
        self.linear_mu_cr = nn.Linear(cellsize_cr, NUM_CR_MEAN)  # output necessary CR mu's
        self.linear_sigma_cr = nn.Linear(cellsize_cr, 1)

        self.lstm_w = nn.LSTMCell(POP_SIZE + BINS * 2, cellsize_w)  # (input size, hidden size, bias=True)
        self.linear_mu_w = nn.Linear(cellsize_w, NUM_W_MEAN)
        self.linear_sigma_w = nn.Linear(cellsize_w, 1)

    def forward(self, x_f, h_f, c_f, x_cr, h_cr, c_cr, x_w, h_w, c_w):
        h_next_f, c_next_f = self.lstm_f(x_f, (h_f, h_f))  # x:[seq_len, batch, inputsize]
        mu_f = torch.sigmoid(self.linear_mu_f(h_next_f))  # [batch, hidden]
        sigma_f = torch.sigmoid(self.linear_sigma_f(h_next_f))  # [batch, 1]

        h_next_cr, c_next_cr = self.lstm_cr(x_cr, (h_cr, c_cr))
        mu_cr = torch.sigmoid(self.linear_mu_cr(h_next_cr))
        sigma_cr = torch.sigmoid(self.linear_sigma_cr(h_next_cr))

        h_next_w, c_next_w = self.lstm_w(x_w, (h_w, c_w))
        mu_w = torch.sigmoid(self.linear_mu_w(h_next_w))
        sigma_w = torch.sigmoid(self.linear_sigma_w(h_next_w))
        return mu_f, sigma_f, h_next_f, c_next_f, mu_cr, sigma_cr, h_next_cr, c_next_cr, mu_w, sigma_w, h_next_w, c_next_w

    def sampler(self, x_f, h_f, c_f, x_cr, h_cr, c_cr, x_w, h_w, c_w):
        mu_f, sigma_f, h_next_f, c_next_f, mu_cr, sigma_cr, h_next_cr, c_next_cr, mu_w, sigma_w, h_next_w, c_next_w = \
            self.forward(x_f, h_f, c_f, x_cr, h_cr, c_cr, x_w, h_w, c_w)
        normal_f = torch.distributions.Normal(mu_f, sigma_f) #mu[1, num_f_mu+num_cr_mu], sigma[1,1]=[[0.1]]
        sample_f = torch.clamp(normal_f.rsample(), 0, 1)
        normal_cr = torch.distributions.Normal(mu_cr, sigma_cr)  # mu[1, num_f_mu+num_cr_mu], sigma[1,1]=[[0.1]]
        sample_cr = torch.clamp(normal_cr.rsample(), 0, 1)

        if mu_w.size(1) == POP_SIZE:
            normal_w1 = torch.distributions.Normal(torch.squeeze(mu_w)[0:P], torch.squeeze(sigma_w))
            sample_w1 = torch.clamp(normal_w1.rsample([P-1]), 0, 1)  # [P-1, P]
            normal_w2 = torch.distributions.Normal(torch.squeeze(mu_w)[P:POP_SIZE], torch.squeeze(sigma_w))
            sample_w2 = torch.clamp(normal_w2.rsample([P]), 0, 1)   # [P, N-P]
            sample_w = [sample_w1, sample_w2]
        elif mu_w.size(1) == P:
            normal_w = torch.distributions.Normal(torch.squeeze(mu_w), torch.squeeze(sigma_w))
            sample_w = torch.clamp(normal_w.rsample([POP_SIZE]), 0, 1)  # change in V3, see doc.

        return sample_f, h_next_f, c_next_f, sample_cr, h_next_cr, c_next_cr, sample_w, h_next_w, c_next_w


class LdePop(object):  # input the population information
    def __init__(self, cellsize_f, cellsize_cr, cellsize_w, learning_rate):
        self.hidden_f = cellsize_f
        self.hidden_cr = cellsize_cr
        self.hidden_w = cellsize_w
        self.lde_net = PGAgent(cellsize_f, cellsize_cr, cellsize_w)
        self.optimizer = torch.optim.Adam(self.lde_net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, last_epoch=-1)
        self.inputs_f, self.hs_f, self.cs_f = [], [], []
        self.inputs_cr, self.hs_cr, self.cs_cr, self.inputs_w, self.hs_w, self.cs_w, = [], [], [], [], [], []
        self.obs, self.sort_ind, self.sort_ind_last = [], [], []
        self.fs, self.crs, self.ws, self.rewards1, self.rewards2 = [], [], [], [], []
        self.lde_net.to(device)

    def fix(self, x_f, h_f, c_f, x_cr, h_cr, c_cr, x_w, h_w, c_w):
        return self.lde_net.forward(x_f, h_f, c_f, x_cr, h_cr, c_cr, x_w, h_w, c_w)

    def sampler(self, x_f, h_f, c_f, x_cr, h_cr, c_cr, x_w, h_w, c_w):
        return self.lde_net.sampler(x_f, h_f, c_f, x_cr, h_cr, c_cr, x_w, h_w, c_w)

    def store_transition(self, in_f, h_f, c_f, in_cr, h_cr, c_cr, in_w, h_w, c_w, f, cr, w, r1):  # (s,a,r,s_)
        self.inputs_f.append(in_f)
        self.inputs_cr.append(in_cr)
        self.inputs_w.append(in_w)
        # self.obs.append(ob_t)  # ob_list[ob_1^0, ..,ob_l^t,..] where each is [N,ob_size],vertical [P*L*T*N,ob_size]
        self.hs_f.append(h_f)  # list, each [P*L*T*N,hidden_size], first i in N, then t in T, l in L at last
        self.cs_f.append(c_f)
        self.hs_cr.append(h_cr)
        self.cs_cr.append(c_cr)
        self.hs_w.append(h_w)
        self.cs_w.append(c_w)

        self.fs.append(f)  # [P*L*T*N,1]
        self.crs.append(cr)
        self.ws.append(w)  # [P*L*T*(p-1), p]

        self.rewards1.append(r1)  # [P*L*T,N]

    def test_net(self, or_f_mu, or_f_std, or_cr_mu, or_cr_std, or_w_mu, or_w_std):
        self.ors_f_mu.append(or_f_mu)
        self.ors_f_std.append(or_f_std)
        self.ors_cr_mu.append(or_cr_mu)
        self.ors_cr_std.append(or_cr_std)
        self.ors_w_mu.append(or_w_mu)
        self.ors_w_std.append(or_w_std)

    def store_r2(self, r2):
        self.rewards2.append(r2)

    def discount_baseline_trunc_r(self, lambda_global_flag, reward_flag):
        reward2 = maxmin_norm_reward2(self.rewards2)
        reward3 = r3_func(self.rewards1, reward2)

        discount_b_reward1 = discounted_with_baseline_trunc_rewards(self.rewards1, trunc0_flag=RELU_REWARD, norm_flag=NORM_REWARD)  # array[P*L*T]
        discount_b_reward2 = baseline_trunc_expand_reward2(reward2, trunc0_flag=RELU_REWARD, norm_flag=NORM_REWARD)
        discount_b_reward3 = discount_baseline_trunc_rewards(self.rewards1, reward2, lambda_global_flag,
                                                             norm_flag=NORM_REWARD, trunc0_flag=RELU_REWARD)

        # one step update
        if reward_flag == 1:
            dis_b_reward = discount_b_reward1
        elif reward_flag == 2:
            dis_b_reward = discount_b_reward2
        elif reward_flag == 3:
            dis_b_reward = discount_b_reward3
        return dis_b_reward, discount_b_reward1, discount_b_reward2, discount_b_reward3, reward2, reward3

    def learn(self, dbt_reward, device):  # no input, use lists in transitions stored above
        all_mean_f, all_std_f, all_h_f, all_c_f, all_mean_cr, all_std_cr, all_h_cr, all_c_cr, all_mean_w, \
        all_std_w, all_hw, all_cw = self.lde_net.forward(torch.squeeze(torch.stack(self.inputs_f)).to(device),
                                                         torch.squeeze(torch.stack(self.hs_f)).to(device),
                                                         torch.squeeze(torch.stack(self.cs_f)).to(device),
                                                         torch.squeeze(torch.stack(self.inputs_cr)).to(device),
                                                         torch.squeeze(torch.stack(self.hs_cr)).to(device),
                                                         torch.squeeze(torch.stack(self.cs_cr)).to(device),
                                                         torch.squeeze(torch.stack(self.inputs_w)).to(device),
                                                         torch.squeeze(torch.stack(self.hs_w)).to(device),
                                                         torch.squeeze(torch.stack(self.cs_w)).to(device))

        normal_f = torch.distributions.Normal(all_mean_f, all_std_f)  # [20,2]+[20,1]  [P*L*T,2]
        log_prob_f = torch.sum(normal_f.log_prob(torch.squeeze(torch.stack(self.fs, 0)) + 1e-8), 1)
        loss_f = torch.mean(torch.mul(log_prob_f, torch.FloatTensor(dbt_reward).to(device)).mul(-1))

        normal_cr = torch.distributions.Normal(all_mean_cr, all_std_cr)
        log_prob_cr = torch.sum(normal_cr.log_prob(torch.squeeze(torch.stack(self.crs, 0)) + 1e-8), 1)
        loss_cr = torch.mean(torch.mul(log_prob_cr, torch.FloatTensor(dbt_reward).to(device)).mul(-1))

        normal_w = torch.distributions.Normal(all_mean_w, all_std_w)
        log_prob_w = torch.mean(torch.sum(normal_w.log_prob(torch.squeeze(torch.stack(self.ws, 1)) + 1e-8), 2), 0)
        loss_w = torch.mean(torch.mul(log_prob_w, torch.FloatTensor(dbt_reward).to(device)).mul(-1))

        loss = (loss_f + loss_cr + loss_w) / 3.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        torch.cuda.empty_cache()
        return all_std_f, all_std_cr, all_std_w

    def save_model(self, path):
        torch.save(self.lde_net.state_dict(), path, _use_new_zipfile_serialization=False)

    def save_rs(self, filename, epoch, reward_flag):
        if reward_flag == 1:
            np.savetxt(filename + '/rs' + str(reward_flag) + '_Epoch' + str(epoch) + '.txt', self.rewards1.flatten())
        elif reward_flag == 2:
            np.savetxt(filename + '/rs' + str(reward_flag) + '_Epoch' + str(epoch) + '.txt', self.rewards2.flatten())
        elif reward_flag == 3:
            print('r3 not')
            # np.savetxt(filename + '/rs' + str(reward_flag) + '_Epoch' + str(epoch) + '.txt', )
        np.savetxt(filename + '/rs' + str(reward_flag) + '_Epoch' + str(epoch) + '.txt', self.rewards2.flatten())

    def next_pop_start(self):
        pop_ini = np.random.uniform(X_MIN, X_MAX, (POP_SIZE, PROBLEM_SIZE))
        self.inputs_f, self.hs_f, self.cs_f = [], [], []
        self.inputs_cr, self.hs_cr, self.cs_cr, self.inputs_w, self.hs_w, self.cs_w, = [], [], [], [], [], []
        self.obs, self.sort_ind, self.sort_ind_last = [], [], []
        self.fs, self.crs, self.ws, self.rewards1, self.rewards2 = [], [], [], [], []
        return pop_ini

    def load_model(self, path):
        self.lde_net.load_state_dict(torch.load(path))
