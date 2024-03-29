import decimal
import numpy as np
PI = 3.141592653
E = 2.718281828

decimal.getcontext().prec = 100

def extract_odds(r_mat):
    odd_index_column = np.array([0, 2, 4, 6, 8])
    odd_index_row = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
                              40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78,
                              80, 82, 84, 86, 88, 90, 92, 94, 96, 98])
    m1 = r_mat[odd_index_row, :]
    rotate = m1[:, odd_index_column]
    return rotate

def cec17(pop, func_index):
    # print('cec17 func', func_index)
    pop = np.transpose(pop)  # [N.D]->[D,N]
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if pop_dim == 5 and func_index > 20:
        rotate_file = "input_data/M_" + str(func_index) + "_D" + str(pop_dim+5) + ".txt"
    else:
        rotate_file = "input_data/M_" + str(func_index) + "_D" + str(pop_dim) + ".txt"
    shift_file = "input_data/shift_data_" + str(func_index) + ".txt"
    if 10 < func_index < 21 or func_index == 29 or func_index == 30:
        shuffle_file = "input_data/shuffle_data_" + str(func_index) + "_D" + str(pop_dim) + ".txt"
    if pop_dim != 2 and pop_dim != 10 and pop_dim != 5 and pop_dim != 30 and pop_dim != 50 and pop_dim != 100:
        print("The dimension of pop should be 2 5 10 30 50 and 100!!")
    # shift = (np.loadtxt(shift_file)).reshape(-1, 1)
    shift = np.loadtxt(shift_file) #对于后20函数，shift[10,100],cfi的shift阵来自第i行的前D个
    if func_index < 21:
        shift = shift.reshape(-1, 1)
        shift = shift[0:pop_dim]  # [D, 1] #读入shift[100,]
        rotate = (np.loadtxt(rotate_file)).reshape(pop_dim, pop_dim)  # [D,D]
        if func_index > 10:
            shuffle = (np.loadtxt(shuffle_file)).reshape(-1, 1) #[D,1]
    elif func_index == 21 or func_index == 22:
        shift = np.transpose(shift[0:3, 0:pop_dim])
        # shift = np.transpose(shift[0:3*pop_dim].reshape(-1, pop_dim)) #[ND,1]->[D,N],fi的shift阵为[:,i].reshape(-1,1)
        rotate = np.loadtxt(rotate_file)
        if pop_dim == 5:
            rotate = (extract_odds(rotate)).reshape(-1, 1) #for 5 dim, rotate matrix is acquired from D10, odd row and odd column
        else:
            rotate = rotate.reshape(-1, 1)
        # rotate = (np.loadtxt(rotate_file)).reshape(-1, 1)
        rotate = rotate[0:3*pop_dim*pop_dim].reshape(3, pop_dim, pop_dim)#[NDD,1]->[N,D,D],rotatei=[i,:,:]
    elif func_index == 24 or func_index == 23:
        shift = np.transpose(shift[0:4, 0:pop_dim].reshape(-1, pop_dim))  # [ND,1]->[D,N],fi的shift阵为[:,i].reshape(-1,1)
        # rotate = (np.loadtxt(rotate_file)).reshape(-1, 1)
        rotate = np.loadtxt(rotate_file)
        if pop_dim == 5:
            rotate = (extract_odds(rotate)).reshape(-1, 1)
        else:
            rotate = rotate.reshape(-1, 1)
        rotate = rotate[0:4 * pop_dim * pop_dim].reshape(4, pop_dim, pop_dim)
    elif func_index == 26 or func_index == 25:
        shift = np.transpose(shift[0:5, 0:pop_dim].reshape(-1, pop_dim))  # [ND,1]->[D,N],fi的shift阵为[:,i].reshape(-1,1)
        # rotate = (np.loadtxt(rotate_file)).reshape(-1, 1)
        rotate = np.loadtxt(rotate_file)
        if pop_dim == 5:
            rotate = (extract_odds(rotate)).reshape(-1, 1)
        else:
            rotate = rotate.reshape(-1, 1)
        rotate = rotate[0:5 * pop_dim * pop_dim].reshape(5, pop_dim, pop_dim)
    elif func_index == 28 or func_index == 27:
        shift = np.transpose(shift[0:6, 0:pop_dim].reshape(-1, pop_dim))  # [ND,1]->[D,N],fi的shift阵为[:,i].reshape(-1,1)
        # rotate = (np.loadtxt(rotate_file)).reshape(-1, 1)
        rotate = np.loadtxt(rotate_file)
        if pop_dim == 5:
            rotate = (extract_odds(rotate)).reshape(-1, 1)
        else:
            rotate = rotate.reshape(-1, 1)
        rotate = rotate[0:6 * pop_dim * pop_dim].reshape(6, pop_dim, pop_dim)
    elif func_index == 29 or func_index == 30:
        shift = np.transpose(shift[0:3, 0:pop_dim])
        # rotate = (np.loadtxt(rotate_file)).reshape(-1, 1)
        rotate = np.loadtxt(rotate_file)
        if pop_dim == 5:
            rotate = (extract_odds(rotate)).reshape(-1, 1)
        else:
            rotate = rotate.reshape(-1, 1)
        rotate = rotate[0:3 * pop_dim * pop_dim].reshape(3, pop_dim, pop_dim)
        shuffle = (np.loadtxt(shuffle_file)).reshape(-1, 1)
        shuffle = (np.transpose(shuffle[0:pop_dim*3].reshape(3, pop_dim))).astype('int')


    if func_index == 1:
        fitness = bent_cigar_func(pop, shift, rotate, s_flag=1, r_flag=1)
    elif func_index == 2:
        fitness = sum_diff_pow_func(pop, shift, rotate, s_flag=0, r_flag=0)
    elif func_index == 3:
        fitness = zakharov_func(pop, shift, rotate, s_flag=1, r_flag=1)
    elif func_index == 4:
        fitness = rosenbrock_func(pop, shift, rotate, s_flag=1, r_flag=1)
    elif func_index == 5:
        fitness = rastrigin_func(pop, shift, rotate, s_flag=1, r_flag=1)
    elif func_index == 6:
        fitness = schaffer_F7_func(pop, shift, rotate, s_flag=1, r_flag=1)
    elif func_index == 7:
        fitness = bi_rastrigin_func(pop, shift, rotate, s_flag=1, r_flag=1)
    elif func_index == 8:
        fitness = step_rastrigin_func(pop, shift, rotate, s_flag=1, r_flag=1)
    elif func_index == 9:
        fitness = levy_func(pop, shift, rotate, s_flag=1, r_flag=1) #pdf公式乘项应为5.12/100
    elif func_index == 10:
        fitness = modif_schwefel_func(pop, shift, rotate, s_flag=1, r_flag=1)
    elif func_index == 11:
        fitness = hf01(pop, shift, rotate, shuffle, s_flag=1, r_flag=1)
    elif func_index == 12:
        fitness = hf02(pop, shift, rotate, shuffle, s_flag=1, r_flag=1)
    elif func_index == 13:
        fitness = hf03(pop, shift, rotate, shuffle, s_flag=1, r_flag=1)
    elif func_index == 14:
        fitness = hf04(pop, shift, rotate, shuffle, s_flag=1, r_flag=1)+4.590461344378127e-10
    elif func_index == 15:
        fitness = hf05(pop, shift, rotate, shuffle, s_flag=1, r_flag=1)
    elif func_index == 16:
        fitness = hf06(pop, shift, rotate, shuffle, s_flag=1, r_flag=1)
    elif func_index == 17:
        fitness = hf07(pop, shift, rotate, shuffle, s_flag=1, r_flag=1)
    elif func_index == 18:
        fitness = hf08(pop, shift, rotate, shuffle, s_flag=1, r_flag=1)+4.590461344378127e-10
    elif func_index == 19:
        fitness = hf09(pop, shift, rotate, shuffle, s_flag=1, r_flag=1) # -8.973275017663613e-14
    elif func_index == 20:
        fitness = hf10(pop, shift, rotate, shuffle, s_flag=1, r_flag=1)+4.590461344378127e-10
    elif func_index == 21:
        fitness = cf01(pop, shift, rotate, r_flag=1)
    elif func_index == 22:
        fitness = cf02(pop, shift, rotate, r_flag=1)
    elif func_index == 23:
        fitness = cf03(pop, shift, rotate, r_flag=1)
    elif func_index == 24:
        fitness = cf04(pop, shift, rotate, r_flag=1)+4.590461344378127e-10
    elif func_index == 25:
        fitness = cf05(pop, shift, rotate, r_flag=1)
    elif func_index == 26:
        fitness = cf06(pop, shift, rotate, r_flag=1)
    elif func_index == 27:
        fitness = cf07(pop, shift, rotate, r_flag=1)
    elif func_index == 28:
        fitness = cf08(pop, shift, rotate, r_flag=1)
    elif func_index == 29:
        fitness = cf09(pop, shift, rotate, shuffle, r_flag=1)
    elif func_index == 30:
        fitness = cf10(pop, shift, rotate, shuffle, r_flag=1)
    return fitness

def bent_cigar_func(pop, shift, rotate, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    for j in range(pop_size):
        f = pop_sr[0, j]*pop_sr[0, j]
        for i in range(1, pop_dim):
            f += 1e6*pop_sr[i, j]*pop_sr[i, j]
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def shift_rotate_pop(pop, s_mat, r_mat, sh_rate):
    pop_size = pop.shape[1]
    for i in range(pop_size):
        if i == 0:
            shift_o = s_mat
        else:
            shift_o = np.hstack((shift_o, s_mat))  #D*N
    pop_sr = np.dot(r_mat, sh_rate*(pop-shift_o))
    return pop_sr

def shift_pop(pop, s_mat):
    pop_size = pop.shape[1]
    for i in range(pop_size):
        if i == 0:
            shift_o = s_mat
        else:
            shift_o = np.hstack((shift_o, s_mat))
    pop_shift = pop-shift_o
    return pop_shift

def bi_rastrigin_func(pop, shift, rotate, s_flag, r_flag): # Lunacek Bi_rastrigin Function No.7
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    mu0 = 2.5
    d = 1.0
    s = 1.0-1.0/(2.0*pow(pop_dim+20.0, 0.5)-8.2)
    # s = 1.0 - 1.0 / (2.0*pow(pop_dim+20.0, 0.5)-8.2)
    mu1 = -pow((mu0*mu0-d)/s, 0.5)
    if s_flag == 1:
        pop_s = shift_pop(pop, shift)
    else:
        pop_s = pop
    pop_s *= 10.0/100.0
    tmpx = 2*pop_s
    for j in range(pop_size):
        for i in range(pop_dim):
            if shift[i] < 0.0:
                tmpx[i, j] *= -1.0

    z = tmpx.copy()
    tmpx += mu0

    # tmp1 = np.zeros((1, pop_size))
    # tmp2 = np.zeros((1, pop_size))
    # for j in range(pop_size):
    #     for i in range(pop_dim):
    #         tmp = tmpx[i, j] - mu0
    #         tmp1[0, j] += tmp*tmp
    #         tmp = tmpx[i, j] - mu1
    #         tmp2[0, j] += tmp*tmp
    #     tmp2[0, j] *= s
    #     tmp2[0, j] += d*pop_dim
    # if r_flag == 1:
    #     pop_r = np.dot(rotate, z)
    # else:
    #     pop_r = z
    # for j in range(pop_size):
    #     tmp = 0.0
    #     for i in range(pop_dim):
    #         tmp += np.cos(2.0*PI*pop_r[i, j])
    #     if tmp1[0, j] < tmp2[0, j]:
    #         f = tmp1[0, j]
    #     else:
    #         f = tmp2[0, j]
    #     f += 10.0*(pop_dim-tmp)
    #     if j == 0:
    #         fit = f
    #     else:
    #         fit = np.vstack((fit, f))

    if r_flag == 1:
        pop_r = np.dot(rotate, z)
    else:
        pop_r = z

    for j in range(pop_size):
        tmp1 = 0.0
        tmp2 = 0.0
        for i in range(pop_dim):
            tmp = tmpx[i, j] - mu0
            tmp1 += tmp*tmp
            tmp = tmpx[i, j] - mu1
            tmp2 += tmp*tmp
        tmp2 *= s
        tmp2 += d*pop_dim
        tmp = 0.0
        for i in range(pop_dim):
            tmp += np.cos(2.0*PI*pop_r[i, j])
        if tmp1 < tmp2:
            f = tmp1
        else:
            f = tmp2
        f += 10.0*(pop_dim-tmp)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def step_rastrigin_func(pop, shift, rotate, s_flag, r_flag):  #Noncontinuous Rastrigin's No.8
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 5.12/100.0)
    else:
        pop_sr = pop * 5.12/100.0
    for j in range(pop_size):
        f = 0
        for i in range(pop_dim):
            # if np.fabs(pop_sr[i, j] - shift[i] > 0.5):
            #     # pop_sr[i, j] = np.floor(2*pop_sr[i, j])/2
            #     pop_sr[i, j] = shift[i] + np.floor(2*(pop_sr[i, j]-shift[i])+0.5)/2
            f += (pop_sr[i, j]*pop_sr[i, j] - 10.0*np.cos(2.0*PI*pop_sr[i, j]) + 10.0)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def levy_func(pop, shift, rotate, s_flag, r_flag):  #No.9
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    w = 1.0+(pop_sr-1.0)/4.0
    for j in range(pop_size):
        term1 = pow((np.sin(PI*w[0, j])), 2)
        term2 = pow((w[pop_dim-1, j]-1), 2)*(1+pow((np.sin(2*PI*w[pop_dim-1, j])), 2))
        sum = 0.0
        for i in range(pop_dim-1):
            newv = pow((w[i, j]-1), 2)*(1+10*pow((np.sin(PI*w[i, j]+1)), 2))
            sum += newv
        f = term1 + sum + term2
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def modif_schwefel_func(pop, shift, rotate, s_flag, r_flag):  #Modified Schwefel’s No.10
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    # fit = np.zeros((pop_size, 1))
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1000.0/100.0)
    else:
        pop_sr = pop * 1000.0/100.0
    z = pop_sr + 4.209687462275036e+002
    for j in range(pop_size):
        f = 0
        for i in range(pop_dim):
            if z[i, j] > 500:
                f -= (500.0-np.mod(z[i, j], 500))*np.sin(pow(500.0-np.mod(z[i, j], 500), 0.5))
                tmp = (z[i, j]-500.0)/100
                f += tmp*tmp/pop_dim
            elif z[i, j] < -500:
                f -= (-500.0+np.mod(np.fabs(z[i, j]), 500))*np.sin(pow(500.0-np.mod(np.fabs(z[i, j]), 500), 0.5))
                tmp = (z[i, j]+500.0)/100
                f += tmp*tmp/pop_dim
            else:
                f -= z[i, j]*np.sin(pow(np.fabs(z[i, j]), 0.5))
        f += 4.189828872724338e+002*pop_dim
        # fit[j, 0] = f
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def ellips_func(pop, shift, rotate, s_flag, r_flag):   #No.11, High Conditioned Elliptic
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    for j in range(pop_size):
        f = 0
        for i in range(pop_dim):
            if pop_dim == 1:
                f += pow(10.0, 6.0 * i / pop_dim) * pop_sr[i, j] * pop_sr[i, j]
            else:
                f += pow(10.0, 6.0*i/(pop_dim-1))*pop_sr[i, j]*pop_sr[i, j]
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def sum_diff_pow_func(pop, shift, rotate, s_flag, r_flag):  #No.2
    pop_dim = pop.shape[0]  # pop[D,N]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    for j in range(pop_size):
        sum = 0.0
        for i in range(pop_dim):
            newv = pow(np.abs(pop_sr[i, j]), float(i+2)) #pow(np.abs,)会计算出错,pow中参数需为float类型
            sum = sum + newv
        f = sum
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def discus_func(pop, shift, rotate, s_flag, r_flag): #No.12
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    for j in range(pop_size):
        f = pow(10.0, 6.0)*pop_sr[0, j]*pop_sr[0, j]
        for i in range(pop_dim):
            if i > 0:
                f += pop_sr[i, j]*pop_sr[i, j]
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

# def sum_diff_pow_func(pop, shift, rotate, s_flag, r_flag): #
#     pop_dim = pop.shape[0]
#     pop_size = pop.shape[1]
#     if s_flag == 1 and r_flag == 1:
#         pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
#     else:
#         pop_sr = pop
#     for j in range(pop_size):
#         f = 0
#         for i in range(pop_dim):
#             f += pow(np.abs(pop_sr[i, j]), i+1)
#         if j == 0:
#             fit = f
#         else:
#             fit = np.vstack((fit, f))
#     return fit

def schaffer_F7_func(pop, shift, rotate, s_flag, r_flag):  #No.20
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        # pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
        pop_sr = shift_pop(pop, shift) #demo code中对x仅平移，不缩放不旋转
    else:
        pop_sr = pop
    # pop_sr [:, 0] = [71.33, 31.06,49.45,23.44,-20.42,61.68,-66.02,-6.63,64.84,10.03]
    # pop_sr[:, 0] = [71.33, 31.06]
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim-1):
            s = pow(pop_sr[i, j]*pop_sr[i, j]+pop_sr[i+1, j]*pop_sr[i+1, j], 0.5)
            # f += pow(s, 0.5)*(np.sin(50*pow(s, 0.2))+1) pdf公式
            tmp = np.sin(50*pow(s, 0.2))
            f += pow(s, 0.5)+pow(s, 0.5)*tmp*tmp
        if pop_dim == 1:
            f = pow(f / pop_dim, 2.0)
        else:
            f = pow(f/(pop_dim-1), 2.0)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def ackley_func(pop, shift, rotate, s_flag, r_flag):  #No.13
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    for j in range(pop_size):
        sum1 = 0.0
        sum2 = 0.0
        for i in range(pop_dim):
            sum1 += pop_sr[i, j]*pop_sr[i, j]
            sum2 += np.cos(2.0*PI*pop_sr[i, j])
        sum1 = -0.2*np.sqrt(sum1/pop_dim)
        sum2 /= pop_dim
        f = E - 20.0*np.exp(sum1)-np.exp(sum2)+20.0
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def weierstrass_func(pop, shift, rotate, s_flag, r_flag):  #No.14
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, decimal.Decimal(0.5/100))
    else:
        pop_sr = pop * 0.5/100.0
    for j in range(pop_size):
        a = 0.5
        b = 3.0
        k_max = 20
        f = 0
        for i in range(pop_dim):
            sum = 0.0
            sum2 = 0.0
            for k in range(k_max):
                sum += pow(a, k)*np.cos(2.0*PI*pow(b, k)*(pop_sr[i, j]+0.5))
                sum2 += pow(a, k)*np.cos(2.0*PI*pow(b, k)*0.5)
            f += sum
        f -= pop_dim*sum2
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def griewank_func(pop, shift, rotate, s_flag, r_flag):  #No.15
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 600.0/100.0)
    else:
        pop_sr = pop * 600.0/100.0
    for j in range(pop_size):
        s = 0.0
        p = 1.0
        for i in range(pop_dim):
            s += pop_sr[i, j]*pop_sr[i, j]
            p *= np.cos(pop_sr[i, j]/np.sqrt(1.0+i)) #与pdf公式不同，比pdf多了+1
        f = 1.0+s/4000.0-p
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def katsuura_func(pop, shift, rotate, s_flag, r_flag):  #No.16
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 5.0/100.0)
    else:
        pop_sr = pop * 5.0/100.0
    for j in range(pop_size):
        f = 1.0
        tmp3 = pow(1.0*pop_dim, 1.2)
        for i in range(pop_dim):
            temp = 0.0
            for k in range(1, 33):
                tmp1 = pow(2.0, k)
                tmp2 = tmp1*pop_sr[i, j]
                temp += np.fabs(tmp2-np.floor(tmp2+0.5))/tmp1
            f *= pow(1.0+(i+1)*temp, 10.0/tmp3) #比公式多了+1
        tmp1 = 10.0/pop_dim/pop_dim
        f = f*tmp1-tmp1
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def grie_rosen_func(pop, shift, rotate, s_flag, r_flag):   #Griewank-Rosenbrock No.19
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 5.0/100.0)
    else:
        pop_sr = pop * 5.0/100.0
    for j in range(pop_size):
        f = 0.0
        pop_sr[0, j] += 1.0
        for i in range(pop_dim-1):
            pop_sr[i+1, j] += 1.0
            tmp1 = pop_sr[i, j]*pop_sr[i, j]-pop_sr[i+1, j]
            tmp2 = pop_sr[i, j]-1.0
            temp = 100.0*tmp1*tmp1 + tmp2*tmp2
            f += (temp*temp)/4000.0 - np.cos(temp) + 1.0
        tmp1 = pop_sr[pop_dim-1, j]*pop_sr[pop_dim-1, j]-pop_sr[0, j]
        tmp2 = pop_sr[pop_dim-1, j]-1.0
        temp = 100.0*tmp1*tmp1+tmp2*tmp2
        f += (temp*temp)/4000.0 - np.cos(temp) + 1.0
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def escaffer6_func(pop, shift, rotate, s_flag, r_flag):  #Expanded Scaffer’s F6 No.6
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim-1):
            temp1 = np.sin(np.sqrt(pop_sr[i, j]*pop_sr[i, j]+pop_sr[i+1, j]*pop_sr[i+1, j]))
            temp1 = temp1*temp1
            temp2 = 1.0+0.001*(pop_sr[i, j]*pop_sr[i, j]+pop_sr[i+1, j]*pop_sr[i+1, j])
            f += 0.5+(temp1-0.5)/(temp2*temp2)
        temp1 = np.sin(np.sqrt(pop_sr[pop_dim-1, j]*pop_sr[pop_dim-1, j]+pop_sr[0, j]*pop_sr[0, j]))
        temp1 = temp1*temp1
        temp2 = 1.0+0.001*(pop_sr[pop_dim-1, j]*pop_sr[pop_dim-1, j]+pop_sr[0, j]*pop_sr[0, j])
        f += 0.5+(temp1-0.5)/(temp2*temp2)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def happycat_func(pop, shift, rotate, s_flag, r_flag):  #No.17
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 5.0/100.0)
    else:
        pop_sr = pop * 5.0/100.0
    alpha = 1.0/8.0
    for j in range(pop_size):
        r2 = 0.0
        sum_z = 0.0
        for i in range(pop_dim):
            pop_sr[i, j] -= 1.0  #why? 公式中并没有-1
            r2 += pop_sr[i, j]*pop_sr[i, j]
            sum_z += pop_sr[i, j]
        f = pow(np.fabs(r2-pop_dim), 2*alpha) + (0.5*r2 + sum_z)/pop_dim + 0.5
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def hgbat_func(pop, shift, rotate, s_flag, r_flag):   #HGBat No.18
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 5.0 / 100.0)
    else:
        pop_sr = pop * 5.0/100.0
    alpha = 1.0 / 4.0
    for j in range(pop_size):
        r2 = 0.0
        sum_z = 0.0
        for i in range(pop_dim):
            pop_sr[i, j] -= 1.0 #why again? Maybe because original global optimum: [-1,-1,...,-1]
            r2 += pop_sr[i, j]*pop_sr[i, j]
            sum_z += pop_sr[i, j]
        f = pow(np.fabs(pow(r2, 2.0)-pow(sum_z, 2.0)), 2*alpha) + (0.5*r2 + sum_z)/pop_dim + 0.5
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def rastrigin_func(pop, shift, rotate, s_flag, r_flag):  #No.5 Rastrigin’s Function
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 5.12/100)  #pdf该缩放项为1
    else:
        pop_sr = pop*5.12/100.0
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim):
            f += (pop_sr[i, j]*pop_sr[i, j]-10.0*np.cos(2.0*PI*pop_sr[i, j])+10.0)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def zakharov_func(pop, shift, rotate, s_flag, r_flag): #f3
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    for j in range(pop_size):
        sum1 = 0.0
        sum2 = 0.0
        for i in range(pop_dim):
            sum1 += pow(pop_sr[i, j], 2.0)
            sum2 += 0.5*(i+1)*pop_sr[i, j]  #与pdf公式不同，多（i+1）乘项
        f = sum1 + pow(sum2, 2.0) + pow(sum2, 4.0)
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def rosenbrock_func(pop, shift, rotate, s_flag, r_flag): #f4
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 2.048/100.0)
    else:
        pop_sr = pop * 2.048/100.0
    pop_sr[0, :] += 1.0
    for j in range(pop_size):
        f = 0.0
        for i in range(pop_dim-1):
            pop_sr[i+1, j] += 1.0
            tmp1 = pop_sr[i, j]*pop_sr[i, j]-pop_sr[i+1, j]
            tmp2 = pop_sr[i, j]-1.0
            f += 100.0*tmp1*tmp1+tmp2*tmp2
        if j == 0:
            fit = f
        else:
            fit = np.vstack((fit, f))
    return fit

def hf01(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.2, 0.4, 0.4])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = zakharov_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)+rosenbrock_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)+rastrigin_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)
    return fit

def hf02(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.3, 0.3, 0.4])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = ellips_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)+modif_schwefel_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)+bent_cigar_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)
    return fit

def hf03(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.3, 0.3, 0.4])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = bent_cigar_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)+rosenbrock_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)+bi_rastrigin_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)
    return fit

def hf04(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.2, 0.2, 0.2, 0.4])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = ellips_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)\
          +ackley_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)\
          +schaffer_F7_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)\
          +rastrigin_func(pop_sr[index[G3[3]:G3[3]+D3_int[3]], :], shift, rotate, 0, 0)
    return fit

def hf05(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.2, 0.2, 0.3, 0.3])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = bent_cigar_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)\
          +hgbat_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)\
          +rastrigin_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)\
          +rosenbrock_func(pop_sr[index[G3[3]:G3[3]+D3_int[3]], :], shift, rotate, 0, 0)
    return fit

def hf06(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.2, 0.2, 0.3, 0.3])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = escaffer6_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)\
          +hgbat_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)\
          +rosenbrock_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)\
          +modif_schwefel_func(pop_sr[index[G3[3]:G3[3]+D3_int[3]], :], shift, rotate, 0, 0)
    return fit

def hf07(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = katsuura_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)\
          +ackley_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)\
          +grie_rosen_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)\
          +modif_schwefel_func(pop_sr[index[G3[3]:G3[3]+D3_int[3]], :], shift, rotate, 0, 0)\
          +rastrigin_func(pop_sr[index[G3[4]:G3[4]+D3_int[4]], :], shift, rotate, 0, 0)
    return fit

def hf08(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = ellips_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)\
          +ackley_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)\
          +rastrigin_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)\
          +hgbat_func(pop_sr[index[G3[3]:G3[3]+D3_int[3]], :], shift, rotate, 0, 0)\
          +discus_func(pop_sr[index[G3[4]:G3[4]+D3_int[4]], :], shift, rotate, 0, 0)
    return fit

def hf09(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = bent_cigar_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)\
          +rastrigin_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)\
          +grie_rosen_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)\
          +weierstrass_func(pop_sr[index[G3[3]:G3[3]+D3_int[3]], :], shift, rotate, 0, 0)\
          +escaffer6_func(pop_sr[index[G3[4]:G3[4]+D3_int[4]], :], shift, rotate, 0, 0)
    return fit

def hf10(pop, shift, rotate, shuffle, s_flag, r_flag):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    if s_flag == 1 and r_flag == 1:
        pop_sr = shift_rotate_pop(pop, shift, rotate, 1.0)
    else:
        pop_sr = pop
    Gp = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    G3 = np.zeros_like(Gp, dtype="int")
    index = (shuffle-1).flatten()
    index = index.astype('int32')
    # index = [int(x) for x in shuffle.flatten()] #shuffle即是个体元素的索引值
                              #同时获取每个分量函数的分量pop，
                              #需要pop[index[前面所有分量pop总共用了的维数Gi:前者+该分量的维数Di],:]
    D3 = np.ceil(Gp*pop_dim) #[2,4,4]
    D3_int = D3.astype('int32')
    for i in range(len(G3)-1):
        G3[i+1] = D3_int[i] + G3[i]
    fit = hgbat_func(pop_sr[index[G3[0]:G3[0]+D3_int[0]], :], shift, rotate, 0, 0)\
          +katsuura_func(pop_sr[index[G3[1]:G3[1]+D3_int[1]], :], shift, rotate, 0, 0)\
          +ackley_func(pop_sr[index[G3[2]:G3[2]+D3_int[2]], :], shift, rotate, 0, 0)\
          +rastrigin_func(pop_sr[index[G3[3]:G3[3]+D3_int[3]], :], shift, rotate, 0, 0)\
          +modif_schwefel_func(pop_sr[index[G3[4]:G3[4]+D3_int[4]], :], shift, rotate, 0, 0)\
          +schaffer_F7_func(pop_sr[index[G3[0]:G3[0]+D3_int[5]], :], shift, rotate, 0, 0)
    #在demo C code中，f20中用于計算的变量表示为y，而在混合函数中，置换后的种群也用y表示了
    #造成在计算schaffer_F7_func时，进入该函数的种群分量是从置换后种群的第一个个体开始
    #而不是该函数顺位的第9个个体开始。c code中变量重叠
    return fit

def cf01(pop, shift, rotate, r_flag):
    cf_num = 3
    sigma = np.array([10, 20, 30])
    bias = np.array([0, 100, 200])
    f1 = rosenbrock_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], 1, r_flag)
    f2 = ellips_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], 1, r_flag)
    f2 = 10000*f2/1e+10
    f3 = rastrigin_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], 1, r_flag)
    f = np.hstack((f1, f2, f3)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf02(pop, shift, rotate, r_flag):
    cf_num = 3
    sigma = np.array([10, 20, 30])
    bias = np.array([0, 100, 200])
    f1 = rastrigin_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], 1, r_flag)
    f2 = griewank_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], 1, r_flag)
    f2 = 1000*f2/100
    f3 = modif_schwefel_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], 1, r_flag)
    f = np.hstack((f1, f2, f3)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf03(pop, shift, rotate, r_flag):
    cf_num = 4
    sigma = np.array([10, 20, 30, 40])
    bias = np.array([0, 100, 200, 300])
    f1 = rosenbrock_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], 1, r_flag)
    f2 = ackley_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], 1, r_flag)
    f2 = 1000*f2/100
    f3 = modif_schwefel_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], 1, r_flag)
    f4 = rastrigin_func(pop, shift[:, 3].reshape(-1, 1), rotate[3, :, :], 1, r_flag)
    f = np.hstack((f1, f2, f3, f4)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf04(pop, shift, rotate, r_flag):
    cf_num = 4
    sigma = np.array([10, 20, 30, 40])
    bias = np.array([0, 100, 200, 300])
    f1 = ackley_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], 1, r_flag)
    f2 = ellips_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], 1, r_flag)
    f2 = 10000*f2/1e10
    f3 = griewank_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], 1, r_flag)
    f3 = 1000*f3/100
    f4 = rastrigin_func(pop, shift[:, 3].reshape(-1, 1), rotate[3, :, :], 1, r_flag)
    f = np.hstack((f1, f2, f3, f4)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf05(pop, shift, rotate, r_flag):
    cf_num = 5
    sigma = np.array([10, 20, 30, 40, 50])
    bias = np.array([0, 100, 200, 300, 400])
    f1 = rastrigin_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], 1, r_flag)
    f1 = 10000*f1/1e3
    f2 = happycat_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], 1, r_flag)
    f2 = 1000*f2/1000
    f3 = ackley_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], 1, r_flag)
    f3 = 1000*f3/100
    f4 = discus_func(pop, shift[:, 3].reshape(-1, 1), rotate[3, :, :], 1, r_flag)
    f4 = 10000*f4/1e10
    f5 = rosenbrock_func(pop, shift[:, 4].reshape(-1, 1), rotate[4, :, :], 1, r_flag)
    f = np.hstack((f1, f2, f3, f4, f5)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf06(pop, shift, rotate, r_flag):
    cf_num = 5
    sigma = np.array([10, 20, 20, 30, 40])
    bias = np.array([0, 100, 200, 300, 400])
    f1 = escaffer6_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], 1, r_flag)
    f1 = 10000*f1/2e7
    f2 = modif_schwefel_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], 1, r_flag)
    f3 = griewank_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], 1, r_flag)
    f3 = 1000*f3/100
    f4 = rosenbrock_func(pop, shift[:, 3].reshape(-1, 1), rotate[3, :, :], 1, r_flag)
    f5 = rastrigin_func(pop, shift[:, 4].reshape(-1, 1), rotate[4, :, :], 1, r_flag)
    f5 = 10000*f5/1e3
    f = np.hstack((f1, f2, f3, f4, f5)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf07(pop, shift, rotate, r_flag):
    cf_num = 6
    sigma = np.array([10, 20, 30, 40, 50, 60])
    bias = np.array([0, 100, 200, 300, 400, 500])
    f1 = hgbat_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], 1, r_flag)
    f1 = 10000*f1/1000
    f2 = rastrigin_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], 1, r_flag)
    f2 = 10000*f2/1e3
    f3 = modif_schwefel_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], 1, r_flag)
    f3 = 10000*f3/4e3
    f4 = bent_cigar_func(pop, shift[:, 3].reshape(-1, 1), rotate[3, :, :], 1, r_flag)
    f4 = 10000*f4/1e30
    f5 = ellips_func(pop, shift[:, 4].reshape(-1, 1), rotate[4, :, :], 1, r_flag)
    f5 = 10000*f5/1e10
    f6 = escaffer6_func(pop, shift[:, 5].reshape(-1, 1), rotate[5, :, :], 1, r_flag)
    f6 = 10000*f6/2e7
    f = np.hstack((f1, f2, f3, f4, f5, f6)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf08(pop, shift, rotate, r_flag):
    cf_num = 6
    sigma = np.array([10, 20, 30, 40, 50, 60])
    bias = np.array([0, 100, 200, 300, 400, 500])
    f1 = ackley_func(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], 1, r_flag)
    f1 = 1000*f1/100
    f2 = griewank_func(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], 1, r_flag)
    f2 = 1000*f2/100
    f3 = discus_func(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], 1, r_flag)
    f3 = 10000*f3/1e10
    f4 = rosenbrock_func(pop, shift[:, 3].reshape(-1, 1), rotate[3, :, :], 1, r_flag)
    f5 = happycat_func(pop, shift[:, 4].reshape(-1, 1), rotate[4, :, :], 1, r_flag)
    f5 = 1000*f5/1e3
    f6 = escaffer6_func(pop, shift[:, 5].reshape(-1, 1), rotate[5, :, :], 1, r_flag)
    f6 = 10000*f6/2e7
    f = np.hstack((f1, f2, f3, f4, f5, f6)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf09(pop, shift, rotate, shuffle, r_flag):
    cf_num = 3
    sigma = np.array([10, 30, 50])
    bias = np.array([0, 100, 200])
    f1 = hf05(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], shuffle[:, 0], 1, r_flag)
    f2 = hf06(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], shuffle[:, 1], 1, r_flag)
    f3 = hf07(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], shuffle[:, 2], 1, r_flag)
    f = np.hstack((f1, f2, f3)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf10(pop, shift, rotate, shuffle, r_flag):
    cf_num = 3
    sigma = np.array([10, 30, 50])
    bias = np.array([0, 100, 200])
    f1 = hf05(pop, shift[:, 0].reshape(-1, 1), rotate[0, :, :], shuffle[:, 0], 1, r_flag)
    f2 = hf08(pop, shift[:, 1].reshape(-1, 1), rotate[1, :, :], shuffle[:, 1], 1, r_flag)
    f3 = hf09(pop, shift[:, 2].reshape(-1, 1), rotate[2, :, :], shuffle[:, 2], 1, r_flag)
    f = np.hstack((f1, f2, f3)).reshape(-1, cf_num) #[N,1]->[N,3]
    fit = cf_cal(pop, f, shift, sigma, bias, cf_num)
    return fit

def cf_cal(pop, f, shift, sigma, bias, cf_num):
    pop_dim = pop.shape[0]
    pop_size = pop.shape[1]
    w = np.zeros_like(f) #w的shape[N, cf_num]
    norm_w = np.zeros_like(f)
    for i in range(pop_size):
        for j in range(cf_num):
            f[i, j] += bias[j]

    for i in range(pop_size):
        for j in range(cf_num):
            tmp=0.0
            for k in range(pop_dim):
                tmp += (pop[k, i]-shift[k, j])*(pop[k, i]-shift[k, j])
            # tmp = np.sum(np.square(pop[:, i] - shift[:, j]))
            if tmp != 0:
                w[i, j] = pow(1.0 / tmp, 0.5) * np.exp(-tmp / 2.0 / pop_dim / pow(sigma[j], 2.0))
            else:
                zero_index = j
                for j in range(cf_num):
                    if j == zero_index:
                        w[i, j] = 1
                    else:
                        w[i, j] = 0
                break


    w_sum = np.sum(w, 1) #[N,],array list
    w_max = np.max(w, 1)
    for i in range(pop_size):
        if w_max[i] == 0:
            for j in range(cf_num):
                w[i, j] = 1
            w_sum[i] = cf_num
    for i in range(pop_size):
        for j in range(cf_num):
            norm_w[i, j] = w[i, j]/w_sum[i]
    fit = (np.sum(f*norm_w, 1)).reshape(-1, 1)
    return fit
# pop = np.zeros((1, 10))
# pop = np.arange(30.0).reshape(-1, 10)
# pop14 = np.array([[-3.6617526987855101e+01, -7.0776084880705966e+01,6.6341409974890496e+01, -3.9922552015596324e+01, 6.1668273872541391e+01, -2.1321024196837577e+01, 5.7546323750888149e+01, -1.8936718822893084e+01, -2.9433256364163597e+01, 3.5260287502436100e+01]])
# pop18 = np.array([[1.8154450402667024e+01,1.8479176246784164e+00,1.7951517970278630e+01, -4.8837473526986997e+01, -1.1417568070542799e+01, -5.5066561612797230e+01, -6.3701026283065332e+01, -6.3148739479085776e+01, -6.0864424994617899e+01, -6.8390884543019752e+01]])
# pop19 = np.array([[ 7.9291185065773874e+01, 6.9474004613921920e+00,-3.5040445381328674e+01, 3.5234476009027333e+01, 2.5838283722213603e+01,2.4034519244001473e+01,-5.1835066066978065e+01, 8.0446122996716838e+00, 5.7989205819549596e+00,2.2458220917013577e+01]])
# # pop20=np.array([[-3.059057464301,-5.5788039181282763e+01,-2.9574519819381891e+01, 6.1045395648593740e+01, 6.9964665914869244e+01, 5.6140031207418076e+01, 2.5504503075734007e+01, 1.4183193861592187e+01,-2.3060768220136296e+01,-6.2331475830791277e+01]])
# pop24=np.array([[ 6.1770164235388009e+01,-8.2403550782428852e+00,-2.1511735494021202e+01, 1.0662928429740914e+01,-2.8853127679253522e+01, 3.1242073274795931e+01,   7.1837517429067532e+01,-6.7539761135140324e+01,-7.9194126903635507e+01, 2.7532766213422995e+01]])
# print(cec17(pop19, 19))
# pop=np.arange(20).reshape(2,10)
# print(hgbat_func(np.transpose(pop),0,0,0,0))