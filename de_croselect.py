"""""
DE's crossover and selection procedures
"""""
from globalVar import *
from CEC13 import *
from CEC17 import *


def modifyChildwithParent(cross_pop, parent_pop, x_max, x_min):
    for i in range(cross_pop.shape[0]):
        for j in range(cross_pop.shape[1]):
            if cross_pop[i, j] < x_min:
                cross_pop[i, j] = (parent_pop[i, j] + x_min)/2.0
            elif cross_pop[i, j] > x_max:
                cross_pop[i, j] = (parent_pop[i, j] + x_max)/2.0
    return cross_pop


def boundConstraint(cross_pop, x_max, x_min):
    for i in range(cross_pop.shape[0]):
        for j in range(cross_pop.shape[1]):
            if cross_pop[i, j] < x_min:
                cross_pop[i, j] = np.minimum(x_max, 2*x_min - cross_pop[i, j])
            elif cross_pop[i, j] > x_max:
                cross_pop[i, j] = np.maximum(x_min, 2*x_max - cross_pop[i, j])
    return cross_pop


def de_crosselect(pop, m_pop, fit, f_vector, cr_vector, succ_f, succ_cr, nfes, index_func, train_flag, func17_flag):
    n_pop = np.zeros_like(pop)
    n_fit = np.zeros_like(fit)
    cr = np.random.uniform(size=(POP_SIZE, PROBLEM_SIZE)) #N,D
    cr_p = cr.copy()

    for i in range(POP_SIZE):
        cr_p[i, cr_p[i, :] < cr_vector[0, i]] = 0
        cr_p[i, cr_p[i, :] >= cr_vector[0, i]] = 1
    cr_m = (cr_p == False).astype('float')
    for i in range(POP_SIZE):
        j = np.random.randint(PROBLEM_SIZE)
        cr_m[i, j] = 1
    cr_p = (cr_m == False).astype('float')
    cross_pop = cr_p * pop + cr_m * m_pop
    cross_pop = modifyChildwithParent(cross_pop, pop, X_MAX, X_MIN)

    if func17_flag == 0:
        cross_fit = cec13(cross_pop, index_func)
    elif func17_flag == 1 and train_flag == 1:
        cross_fit = cec13(cross_pop, index_func)
    elif func17_flag == 1 and train_flag == 0:
        cross_fit = cec17(cross_pop, index_func)

    nfes += POP_SIZE
    for i in range(POP_SIZE):
        if cross_fit[i] < fit[i]:
            n_fit[i] = cross_fit[i]
            n_pop[i] = cross_pop[i]
            succ_f[i] = f_vector[0, i]
            succ_cr[i] = cr_vector[0, i]
        elif cross_fit[i] == fit[i]:
            n_fit[i] = cross_fit[i]
            n_pop[i] = cross_pop[i]
        else:
            n_fit[i] = fit[i]
            n_pop[i] = pop[i]
    return n_pop, n_fit, nfes, succ_f, succ_cr
