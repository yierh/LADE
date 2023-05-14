TEST17 = 0
TRAIN_ACCR = 1  # update the network when the reward is larger than the best before
EPOCH = 2
POP_SIZE = 10
PROBLEM_SIZE = 10
P_MIN = 0.33
P = int(P_MIN * POP_SIZE)
NUM_F_MEAN = POP_SIZE
NUM_CR_MEAN = POP_SIZE
NUM_W_MEAN = P
X_MAX = 100
X_MIN = -100
# new
F_INI = 0.5  # same as jSO
CR_INI = 0.8
# new end
EPSILON = 1e-8     #epsilon
OPTIMUM = 0
LEARNING_RATE = 0.001
GAMMA = 1
BINS = 5
BIN_MEMORY = 5
TRAJECTORY_NUM = 3
TRAJECTORY_LENGTH = 4
OBSERVATION_SIZE = 17
RELU_REWARD = 0
NORM_REWARD = 1
#
# NUM_RUNS = 10
MAXFE = 150*PROBLEM_SIZE
#
PROBLEM_NUM = 2
TEST_PROBLEM = 3

# NES hyper-parameters
NES_METHOD = 0
ES_POP_SIZE = TRAJECTORY_NUM
NOISE_SIGMA = 0.0001