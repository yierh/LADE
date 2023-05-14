"""""
2022.03.24
Filename: main.py
Paper: X. Liu, J. Sun, Q. Zhang, Z. Wang and Z. Xu, "Learning to Learn Evolutionary Algorithm: A Learnable Differential 
       Evolution," in IEEE Transactions on Emerging Topics in Computational Intelligence, 
       doi: 10.1109/TETCI.2023.3251441.
Author: Xin Liu
"""""

from PGagent_3net import LdePop
from utils import *
from lde_pop import lde_pop_train, lde_pop_test
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.cuda.set_device(1)

REWARD_FUNC = 3  # "1,2,3" as the flag of three reward functions
LAMBDA = 1
CELL_SIZE_F = 12  # hidden sizes of three LSTMs
CELL_SIZE_CR = 12
CELL_SIZE_W = 6

np.random.seed(1)
torch.manual_seed(1)

test_data = "CEC13_" if TEST17 == 0 else "CEC17_"
r = "_r1_" if REWARD_FUNC == 1 else "_r2_" if REWARD_FUNC == 2 else "_rs_"
train = "AccR" if TRAIN_ACCR == 1 else "awys"
num = '_NNP_Rev_' if NUM_F_MEAN == POP_SIZE and NUM_CR_MEAN == POP_SIZE and NUM_W_MEAN == P else '_NNN_' if NUM_F_MEAN == POP_SIZE and NUM_CR_MEAN == POP_SIZE and NUM_W_MEAN == P else 'error'


key_info = str(PROBLEM_SIZE) + 'DN' + str(POP_SIZE) + '_F' + str(CELL_SIZE_F) + 'CR' + str(CELL_SIZE_CR) + 'W' \
           + str(CELL_SIZE_W) + '_' + str(LAMBDA) + r + 'p' + str(P_MIN) + 'L' + str(TRAJECTORY_NUM) + \
           'T' + str(TRAJECTORY_LENGTH) + num + train + '_exp0.9'

if PROBLEM_NUM == 26:
    key_info += '_extra6b'
filename = test_data + key_info
test_info = r
test_info += 'AccR_' if TRAIN_ACCR == 1 else ''

try:
    inspect_hyper(TEST17, REWARD_FUNC, LAMBDA)
    print(filename)
    if not os.path.exists(filename):
        os.mkdir(filename)

    PGNet = LdePop(CELL_SIZE_F, CELL_SIZE_CR, CELL_SIZE_W, LEARNING_RATE)  # load model
    lde_pop_train(PGNet, CELL_SIZE_F, CELL_SIZE_CR, CELL_SIZE_W, REWARD_FUNC, LAMBDA, TEST17, filename, key_info,)
    lde_pop_test(PGNet, CELL_SIZE_F, CELL_SIZE_CR, CELL_SIZE_W, 2, TEST17, 0, filename, test_info)

except MyExcept as i:
    print(i)
