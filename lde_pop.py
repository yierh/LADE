"""""
The model comprised by three LSTM outputs entries in LADE
"""""
from de_croselect import *
from utils import *
import os
import scipy.io as scio

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def lde_pop_train(MN, cellsize_f, cellsize_cr, cellsize_w, reward_flag, lambda_global_flag, cec17_flag, filename, info):
    for p in range(PROBLEM_NUM):
        locals()['seq_bsf_mean_f' + str(p+1)] = []
        locals()['seq_reward1_mean_f' + str(p+1)] = []
        locals()['seq_reward2_mean_f' + str(p+1)] = []
        locals()['seq_reward3_mean_f' + str(p + 1)] = []

        locals()['seq_reward1_mean_dbt_f' + str(p + 1)] = []   # mean reward over trajectories after discounted, baseline and truncation, used in loss function
        locals()['seq_reward2_mean_dbt_f' + str(p + 1)] = []
        locals()['seq_reward3_mean_dbt_f' + str(p + 1)] = []

    _, f_labels = get_func_index(0, train_flag=1, func17_flag=cec17_flag)  # just to get the function name/label
    total_best_r = 0.

    for data in range(EPOCH):
        pop_ini = MN.next_pop_start()
        for p in range(PROBLEM_NUM):
            func, _ = get_func_index(p, train_flag=1, func17_flag=cec17_flag)
            fit_ini = cec13(pop_ini, func)
            mean_fit, mean_reward1, mean_reward2 = 0., 0., 0.
            for l in range(TRAJECTORY_NUM):
                pop = pop_ini.copy()
                fit = fit_ini.copy()
                nfes = POP_SIZE
                h_f, c_f = torch.zeros(1, cellsize_f, device=device), torch.zeros(1, cellsize_f, device=device)
                h_cr, c_cr = torch.zeros(1, cellsize_cr, device=device), torch.zeros(1, cellsize_cr, device=device)
                h_w, c_w = torch.zeros(1, cellsize_w, device=device), torch.zeros(1, cellsize_w, device=device)
                s_f, s_cr = F_INI * np.ones(POP_SIZE), CR_INI * np.ones(POP_SIZE)  # successful Fs, CRs
                past_histo = (POP_SIZE / BINS) * np.ones((1, BINS))
                bsf = np.min(fit)
                stag_n = 0
                for t in range(TRAJECTORY_LENGTH):
                    pop, fit = order_by_f(pop, fit)
                    fitness = maxmin_norm(fit)
                    input_f = np.hstack((s_f.reshape(1, -1), np.max(s_f).reshape(1, -1),
                                         np.min(s_f).reshape(1, -1), np.mean(s_f).reshape(1, -1),
                                         np.std(s_f).reshape(1, -1)))
                    input_cr = np.hstack((s_cr.reshape(1, -1), np.max(s_cr).reshape(1, -1),
                                          np.min(s_cr).reshape(1, -1), np.mean(s_cr).reshape(1, -1),
                                          np.std(s_cr).reshape(1, -1)))
                    hist_fit, _ = np.histogram(fitness, BINS)  # hist是array[Bins, ]，len=BINs
                    mean_past_histo = np.mean(past_histo, 0)  # array[Bins, ]
                    input_w = np.hstack((fitness.reshape(1, -1), hist_fit.reshape(1, -1),
                                         mean_past_histo.reshape(1, -1)))
                    f, h_f_, c_f_, cr, h_cr_, c_cr_, w, h_w_, c_w_ = MN.sampler(torch.FloatTensor(input_f).to(device),
                                                                                h_f.to(device), c_f.to(device),
                                                                                torch.FloatTensor(input_cr).to(device),
                                                                                h_cr.to(device), c_cr.to(device),
                                                                                torch.FloatTensor(input_w).to(device),
                                                                                h_w.to(device), c_w.to(device))
                    # first carry out the optimization process, then store the state,action and reward
                    f_trial = f.cpu().detach().numpy().copy().reshape(1, -1)  # [1,N]->[N,1]
                    cr_trial = cr.cpu().detach().numpy().copy().reshape(1, -1)  # [1,N]->[N,1]
                    f_mat = con2mat_F(f_trial)
                    w_mat = con2mat_norm_Pw_wo_loop_v3(w.cpu().detach().numpy().copy())  # sample[N-1]times,[N-1,P]->[N.D]
                    mu_pop = add_random(pop + np.dot(np.multiply(f_mat, w_mat), pop), pop, f_trial)

                    pop_next, fit_next, nfes, s_f, s_cr = de_crosselect(pop, mu_pop, fit, f_trial, cr_trial, s_f, s_cr,
                                                                        nfes, func, train_flag=1,
                                                                        func17_flag=cec17_flag)

                    reward1 = (np.min(fit) - np.min(fit_next)) / np.min(fit)
                    mean_reward1 += reward1  # just for plotting 因为γ=1，reward on one episode = G0=Σrt

                    MN.store_transition(torch.FloatTensor(input_f), h_f, c_f, torch.FloatTensor(input_cr), h_cr, c_cr,
                                        torch.FloatTensor(input_w), h_w, c_w, f, cr, w, reward1.flatten())
                    # MN.test_net(or_f_mu, or_f_std, or_cr_mu, or_cr_std, or_w_mu, or_w_std)

                    fit = fit_next.copy()
                    pop = pop_next.copy()
                    h_f, c_f, h_cr, c_cr, h_w, c_w = h_f_, c_f_, h_cr_, c_cr_, h_w_, c_w_
                    if np.min(fit) < bsf:
                        bsf = np.min(fit)
                        stag_n = 0
                    else:
                        stag_n += POP_SIZE
                mean_fit += np.min(fit)
                reward2 = - np.log(np.min(fit))  # ln()
                mean_reward2 += np.mean(reward2)
                MN.store_r2(reward2)
            locals()['seq_bsf_mean_f' + str(p + 1)].append(mean_fit / TRAJECTORY_NUM)
            locals()['seq_reward1_mean_f' + str(p + 1)].append(mean_reward1 / TRAJECTORY_NUM)
            locals()['seq_reward2_mean_f' + str(p + 1)].append(mean_reward2 / TRAJECTORY_NUM)
            print("data:", data + 1, f_labels[p], "initial min fit:", np.min(fit_ini),
                  " mean_bst:", mean_fit / TRAJECTORY_NUM, "mean reward1:", mean_reward1 / TRAJECTORY_NUM,
                  "mean reward2:", mean_reward2 / TRAJECTORY_NUM)
        dis_b_reward, discount_b_reward1, discount_b_reward2, discount_b_reward3, reward2_ori, reward3_ori = MN.discount_baseline_trunc_r(lambda_global_flag, reward_flag)
        # save rewards as .txt files for answering R1
        if data == 0 or data == 9:
            if reward_flag == 1:
                np.savetxt(filename + '/rs' + str(reward_flag) + '_Epoch' + str(data) + '.txt',
                           (np.array(MN.rewards1)).reshape(-1, TRAJECTORY_LENGTH))  # [P*L*T]→[P*L,T]前L行是第一个问题的L条轨迹上的所有奖励
            elif reward_flag == 2:
                np.savetxt(filename + '/rs' + str(reward_flag) + '_Epoch' + str(data) + '.txt',
                           (np.array(MN.rewards2)).reshape(PROBLEM_NUM, -1))  # 根据原始奖励函数算出来的r [P,L]每个问题最后时刻f的奖励，中间时刻补0
            elif reward_flag == 3:
                np.savetxt(filename + '/rs' + str(reward_flag) + '_Epoch' + str(data) + '.txt',
                           (np.array(reward3_ori)).reshape(-1, TRAJECTORY_LENGTH))  # [P*L,T]
            np.savetxt(filename + '/dis_norm_rs' + str(reward_flag) + '_Epoch' + str(data) + '.txt',
                       (np.array(dis_b_reward)).reshape(-1, TRAJECTORY_LENGTH))  # [P*L, T]mean norm后的Gt（r2甚至max min norm）

        for p in range(PROBLEM_NUM):
            locals()['seq_reward1_mean_dbt_f' + str(p + 1)].append(np.sum(discount_b_reward1[p * TRAJECTORY_LENGTH * TRAJECTORY_NUM:(p + 1) * TRAJECTORY_LENGTH * TRAJECTORY_NUM]) / TRAJECTORY_NUM)
            locals()['seq_reward2_mean_dbt_f' + str(p + 1)].append(np.sum(discount_b_reward2[p * TRAJECTORY_LENGTH * TRAJECTORY_NUM:(p + 1) * TRAJECTORY_LENGTH * TRAJECTORY_NUM]) / TRAJECTORY_NUM)
            locals()['seq_reward3_mean_dbt_f' + str(p + 1)].append(np.sum(discount_b_reward3[p * TRAJECTORY_LENGTH * TRAJECTORY_NUM:(p + 1) * TRAJECTORY_LENGTH * TRAJECTORY_NUM]) / TRAJECTORY_NUM)
            locals()['seq_reward3_mean_f' + str(p + 1)].append(np.sum(reward3_ori[p * TRAJECTORY_LENGTH * TRAJECTORY_NUM:(p + 1) * TRAJECTORY_LENGTH * TRAJECTORY_NUM]) / TRAJECTORY_NUM)

        # update the parameters of the network if the total rewards of this cycle (1epoch) is improved
        total_r = np.sum(MN.rewards1) if reward_flag == 1 else np.sum(reward2_ori) if reward_flag == 2 else np.sum(reward3_ori)
        # print(total_r)
        if TRAIN_ACCR == 1:
            if total_r > total_best_r:
                print('Reward improved from {:.4f} to {:.4f}, update the network***'.format(total_best_r, total_r))
                total_best_r = total_r
                all_std_f, all_std_cr, all_std_w = MN.learn(dis_b_reward, device)
            else:
                print('reward is not improved, best is {:.4f}, so pass {:.4f}.'.format(total_best_r, total_r))
        else:
            all_std_f, all_std_cr, all_std_w = MN.learn(dis_b_reward, device)

        # plot 3 figures of sigma
        one_sigma_f = np.mean(all_std_f.cpu().detach().numpy().reshape(PROBLEM_NUM, TRAJECTORY_NUM, -1), 1).flatten()
        one_sigma_cr = np.mean(all_std_cr.cpu().detach().numpy().reshape(PROBLEM_NUM, TRAJECTORY_NUM, -1), 1).flatten()
        one_sigma_w = np.mean(all_std_w.cpu().detach().numpy().reshape(PROBLEM_NUM, TRAJECTORY_NUM, -1), 1).flatten()
        if data == 0:
            sigma_f = one_sigma_f
            sigma_cr = one_sigma_cr
            sigma_w = one_sigma_w
        else:
            sigma_f = np.hstack((sigma_f, one_sigma_f))
            sigma_cr = np.hstack((sigma_cr, one_sigma_cr))
            sigma_w = np.hstack((sigma_w, one_sigma_w))
        torch.cuda.empty_cache()
        MN.scheduler.step()  # apply the decrease ot the lr
        # lr_list.append(MN.scheduler.get_lr()[0])

    # print(lr_list)
    # pl.plot(np.arange(1, EPOCH+1, 1), lr_list, clip_on=True)
    # pl.show()
    print("PG done")
    model_label = '13fs_' if cec17_flag == 0 else '17all_'
    MN.save_model(os.path.abspath('.') + "/" + filename + "/pg_net_" + model_label + info)

    x = np.linspace(0, EPOCH - 1, EPOCH)
    for p in range(PROBLEM_NUM):
        f_info = f_labels[p] + '_r' + str(reward_flag)
        f_info += '_AccR' if TRAIN_ACCR == 1 else ''
        plot_fit_2reward(p, x, locals()['seq_bsf_mean_f' + str(p + 1)], locals()['seq_reward1_mean_f' + str(p + 1)],
                         locals()['seq_reward2_mean_f' + str(p + 1)], locals()['seq_reward1_mean_dbt_f' + str(p + 1)],
                         locals()['seq_reward2_mean_dbt_f' + str(p + 1)],
                         locals()['seq_reward3_mean_f' + str(p + 1)], filename, f_info)
        # plot_fit_reward(p, x, locals()['seq_bsf_mean_f' + str(p + 1)], locals()['seq_reward1_mean_f' + str(p + 1)],
        #                                 locals()['seq_reward2_mean_f' + str(p+1)], filename, cec17_flag=cec17_flag)
    x_sigma = np.linspace(0, EPOCH * PROBLEM_NUM * TRAJECTORY_LENGTH - 1, EPOCH * PROBLEM_NUM * TRAJECTORY_LENGTH)
    plot_3sigmas_f_cr_w(x_sigma, sigma_f, sigma_cr, sigma_w, filename)
    print("figures of training saved")


@torch.no_grad()
def lde_pop_test(MN, cellsize_f, cellsize_cr, cellsize_w, runs, cec17_flag, fix_flag, filename, info):
    if fix_flag == 1:
        print("\ntest fix starts")
    else:
        print("\ntest sample starts")
    bsf_array = np.zeros(runs)
    bsf_T_array = np.zeros(runs)
    all_bsf = np.zeros((TEST_PROBLEM, runs))
    all_bsf_T = np.zeros((TEST_PROBLEM, runs))

    for test_p in range(TEST_PROBLEM):
        func, _ = get_func_index(test_p, train_flag=0, func17_flag=cec17_flag)

        for repeat in range(runs):
            locals()['seq_f' + str(test_p + 1)], locals()['seq_cr' + str(test_p + 1)] = [], []
            locals()['seq_w' + str(test_p + 1)] = {}  # 只记录每个问题第1次evolution过程结果
            test_pop = np.random.uniform(X_MIN, X_MAX, (POP_SIZE, PROBLEM_SIZE))
            if cec17_flag == 0:
                test_fit = cec13(test_pop, func)
            else:
                test_fit = cec17(test_pop, func)
            test_nfes = POP_SIZE
            test_h_f = torch.zeros(1, cellsize_f, device=device)
            test_c_f = torch.zeros(1, cellsize_f, device=device)
            test_h_cr = torch.zeros(1, cellsize_cr, device=device)
            test_c_cr = torch.zeros(1, cellsize_cr, device=device)
            test_h_w = torch.zeros(1, cellsize_w, device=device)
            test_c_w = torch.zeros(1, cellsize_w, device=device)
            test_s_f, test_s_cr = F_INI * np.ones(POP_SIZE), CR_INI * np.ones(POP_SIZE)
            test_past_histo = (POP_SIZE / BINS) * np.ones((1, BINS))

            for t in range(int((MAXFE - POP_SIZE) / POP_SIZE)):
                if t == 0:
                    pop_ = test_pop.copy()
                    fit_ = test_fit.copy()
                pop_, fit_ = order_by_f(pop_, fit_)
                fitness_ = maxmin_norm(fit_)
                test_input_f = np.hstack((test_s_f.reshape(1, -1), np.max(test_s_f).reshape(1, -1),
                                          np.min(test_s_f).reshape(1, -1), np.mean(test_s_f).reshape(1, -1),
                                          np.std(test_s_f).reshape(1, -1)))
                test_input_cr = np.hstack((test_s_cr.reshape(1, -1), np.max(test_s_cr).reshape(1, -1),
                                           np.min(test_s_cr).reshape(1, -1), np.mean(test_s_cr).reshape(1, -1),
                                           np.std(test_s_cr).reshape(1, -1)))
                hist_fit_, _ = np.histogram(fitness_, BINS)
                test_mean_past_histo = np.mean(test_past_histo, 0)
                test_input_w = np.hstack((fitness_.reshape(1, -1), hist_fit_.reshape(1, -1),
                                          test_mean_past_histo.reshape(1, -1)))
                if fix_flag == 1:
                    f_mean, f_sigma, test_h_f, test_c_f, cr_mean, cr_sigma, test_h_cr, test_c_cr, w_mean, w_sigma, \
                    test_h_w, test_c_w = MN.fix(torch.FloatTensor(test_input_f).to(device), test_h_f.to(device),
                                                test_c_f.to(device),
                                                torch.FloatTensor(test_input_cr).to(device), test_h_cr.to(device),
                                                test_c_cr.to(device),
                                                torch.FloatTensor(test_input_w).to(device), test_h_w.to(device),
                                                test_c_w.to(device))

                    f_trial_ = f_mean.cpu().detach().numpy().copy().reshape(1, -1)  # [1,N]
                    cr_trial_ = cr_mean.cpu().detach().numpy().copy().reshape(1, -1)
                    w_trial_ = np.tile(np.squeeze(w_mean.cpu().detach().numpy().copy()), [POP_SIZE - 1, 1])
                    w_mat = con2mat_norm_Pw_wo_loop(w_trial_)
                else:
                    f_, test_h_f, test_c_f, cr_, test_h_cr, test_c_cr, w_, test_h_w, test_c_w = MN.sampler(
                        torch.FloatTensor(test_input_f).to(device), test_h_f.to(device), test_c_f.to(device),
                        torch.FloatTensor(test_input_cr).to(device), test_h_cr.to(device), test_c_cr.to(device),
                        torch.FloatTensor(test_input_w).to(device), test_h_w.to(device), test_c_w.to(device))

                    f_trial_ = f_.cpu().detach().numpy().copy().reshape(1, -1)  # [1,N]
                    cr_trial_ = cr_.cpu().detach().numpy().copy().reshape(1, -1)
                    w_mat = con2mat_norm_Pw_wo_loop_v3(w_.cpu().detach().numpy().copy())

                # raw outputs
                locals()['seq_f' + str(test_p + 1)].append(f_trial_)
                locals()['seq_cr' + str(test_p + 1)].append(cr_trial_)
                locals()['seq_w' + str(test_p + 1)].update({'t' + str(t + 1): w_mat})  # list[[P,], [P,],...]每个[P,]记录w[:,0]的均值到w[:,P]的均值
                locals()['seq_w' + str(test_p + 1)].update({'t' + str(t + 1): w_.cpu().detach().numpy().copy()})

                f_mat = con2mat_F(f_trial_)
                mu_pop_ = add_random(pop_ + np.dot(np.multiply(f_mat, w_mat), pop_), pop_, f_trial_)

                pop_next_, fit_next_, test_nfes, test_s_f, test_s_cr = de_crosselect(pop_, mu_pop_, fit_,
                                                                                     f_trial_, cr_trial_,
                                                                                     test_s_f, test_s_cr,
                                                                                     test_nfes, func,
                                                                                     train_flag=0,
                                                                                     func17_flag=cec17_flag)
                pop_ = pop_next_.copy()
                fit_ = fit_next_.copy()
                bsf_test = min(fit_)
                if t + 1 == TRAJECTORY_LENGTH:
                    bsf_T_test = min(fit_)
                if np.min(fit_) < EPSILON: break
            bsf_array[repeat] = bsf_test
            bsf_T_array[repeat] = bsf_T_test
            torch.cuda.empty_cache()
            if cec17_flag == 0:
                print("\rCEC13 func {} run {} done".format(func, repeat + 1), end="")
            else:
                print("\rCEC17 func {} run {} done".format(func, repeat + 1), end="")
            if repeat == 0:
            #     scio.savemat(filename + '/CEC13' + info + 'f' + str(func) + '_F.mat',
            #                  {'f': np.vstack(locals()['seq_f' + str(test_p + 1)])})
            #     scio.savemat(filename + '/CEC13' + info + 'f' + str(func) + '_CR.mat',
            #                  {'cr': np.vstack(locals()['seq_cr' + str(test_p + 1)])})
                scio.savemat(filename + '/CEC13' + info + 'f' + str(func) + '_Wori.mat',
                             locals()['seq_w' + str(test_p + 1)])
        all_bsf[test_p, :] = bsf_array
        all_bsf_T[test_p, :] = bsf_T_array

    test_data = "CEC13_fs_" if cec17_flag == 0 else "CEC17_"
    method = "fix" if fix_flag == 1 else "sap"

    # np.savetxt(filename + '/' + test_data + method + info + 'PG_Rev_MAXFE_41runs.txt', all_bsf)

