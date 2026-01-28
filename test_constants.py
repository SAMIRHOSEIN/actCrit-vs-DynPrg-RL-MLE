from turtle import st
import numpy as np
from scipy.stats import norm

from element.problem_setup import ncs as NCS
from element.problem_setup import na as NA
from element.problem_setup import unit_costs, cs_pfs

from torchrl.envs.utils import ExplorationType

# # region: constants for ele_exp_const.py =====================================
# ELE_CONST_HORIZON = 5
# # ELE_CONST_N_HORIZON = 1
# ELE_CONST_N_EPISODES = 1 # modified to avoid confusion
# ELE_CONST_MAX_COST = 1.0

# # ELE_CONST_RESET_PROB = None
# # ELE_CONST_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# # ELE_CONST_RANDOM_STATE = 24
# ELE_CONST_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# ELE_CONST_DIRICHLET_ALPHA = None
# ELE_CONST_RANDOM_STATE = 'off'

# ELE_CONST_ACTION = 1

# ELE_CONST_EXPLORE_TYPE = ExplorationType.RANDOM
# # endregion ==============================================================


# # region: constants for ele_exp_custom.py ====================================
# ELE_CUSTOM_HORIZON = 5
# # ELE_CUSTOM_N_HORIZON = 1
# ELE_CUSTOM_N_EPISODES = 1 # modified to avoid confusion
# ELE_CUSTOM_MAX_COST = 1.0

# # ELE_CUSTORM_RESET_PROB = None
# # ELE_CUSTORM_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# # ELE_CUSTORM_RANDOM_STATE = 24
# ELE_CUSTOM_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# ELE_CUSTOM_DIRICHLET_ALPHA = None
# ELE_CUSTOM_RANDOM_STATE = 'off'


# ELE_CUSTOM_EXPLORE_TYPE = ExplorationType.RANDOM
# # endregion ==============================================================


# # region: constants for ele_ppo_training.py ==================================
# # env parameters
# ELE_PPO_HORIZON = 5  #20 #5 #20 #5 #75

# ELE_PPO_INC_STEP = True
# ELE_PPO_MAX_COST = 1 #unit_costs.max()


# ELE_PPO_RESET_PROB = None
# ELE_PPO_DIRICHLET_ALPHA = [0.05594704, 0.16108377, 0.05494736, 0.03863813] # For all states: [0.15481776, 0.07666929, 0.04912562, 0.03946825] # Alpha for girder beam # [0.28965147, 0.07418968, 0.04705171, 0.04048269]:alpha for three prestressed elements  # 0.5*np.ones(NCS)
# ELE_PPO_RANDOM_STATE = 42
# # ELE_PPO_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0]) #np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # ELE_PPO_DIRICHLET_ALPHA = None
# # ELE_PPO_RANDOM_STATE = 'off'

# # To train nural network actor choose 'nn' after in evaluation region choose 'nn' for ELE_PPO_EVAL_ACTOR
# # To train soft tree actor choose 'st' after in evaluation region choose 'st' for ELE_PPO_EVAL_ACTOR
# # To train frozen tree actor choose 'st' after in evaluation region choose 'ft' for ELE_PPO_EVAL_ACTOR
# actor_model = 'st'  # 'st', 'nn' soft tree or neural network


# if actor_model == 'st':
#     # soft tree parameters
#     depth_soft = 8 #8
#     beta_soft = 1 #1 #1.75 #1000 #1.75 #1.75 #10/7 #20.0
#     batchnorm_soft = False

#     # --- Regularization on soft-tree actor (routing weights only) ---
#     ELE_PPO_REG_TYPE = "none" #"l1"     # options: "none", "l1", "l2", "groupl1"
    
#     if ELE_PPO_REG_TYPE != "none":
#         ELE_PPO_REG_LAMBDA = 1e-12 # regularization strength

# elif actor_model == 'nn':
#     ELE_PPO_REG_TYPE = "none" # I need to define this to avoid error in ele_ppo_training.py(line: import test_constants)

        

# # network parameters(for neural network or soft tree  actor)
# ELE_PPO_TORCH_SEED = 0
# if ELE_PPO_INC_STEP:
#     ELE_PPO_INPUT_DIM = NCS + 1
# else:
#     ELE_PPO_INPUT_DIM = NCS

# if actor_model == 'nn':
#     ELE_PPO_ACTOR_CELLS = 32
#     ELE_PPO_ACTOR_LAYERS = 2

# ELE_PPO_VALUE_CELLS = 32
# ELE_PPO_VALUE_LAYERS = 2

# ELE_PPO_OUTPUT_DIM = NA




# # GAE parameters
# # gamma has to be 1 to avoid double counting gamma in the env
# # lmbda=0 is equivalent to using TD0
# # lmbda=1 is equivalent to using TD1
# # so lmbda should be between 0 and 1
# ELE_PPO_GAE_GAMMA = 1.0
# ELE_PPO_GAE_LAMBDA = 0.95
# ELE_PPO_AVERAGE_GAE = True

# # PPO loss parameters
# ELE_PPO_ENTROPY_EPS = 0.01
# ELE_PPO_CLIP_EPSILON = (1e-3)
# ELE_PPO_CRITIC_COEF = 1.0

# # collector parameters
# ELE_PPO_EPISODES_PER_BATCH = 32     # how many episodes we collect per iteration 
# ELE_PPO_NUM_ITERATIONS = 1024       # how many iteration: collect→optimize cycles we run in total
# ELE_PPO_FRAMES_PER_BATCH = ELE_PPO_HORIZON*ELE_PPO_EPISODES_PER_BATCH
# ELE_PPO_TOTAL_FRAMES = ELE_PPO_FRAMES_PER_BATCH*ELE_PPO_NUM_ITERATIONS
# ELE_PPO_SPLIT_TRAJS = False

# # training parameters
# ELE_PPO_TRAINING_EPOCHS = 50
# ELE_PPO_SUB_BATCH_SIZE = ELE_PPO_HORIZON*32 # actually we consider  one mini-batch
# ELE_PPO_MAX_GRAD_NORM = 1.0
# ELE_PPO_LR = 1e-3
# ELE_PPO_LR_MIN = 1e-5    # lr reduced to lr_min with total_frames // frames_per_batch
# ELE_PPO_EVAL_FREQ = 1



# # There was no set_exploration_type(...) around the collector in training file in the original version. That means 
# # the actor uses the global default exploration type, which for ProbabilisticActor is ExplorationType.RANDOM (i.e., sampling).
# # Training rollouts were stochastic (actions sampled from the Categorical distribution) in original version.
# # PPO training exploration control
# # I added this boolean to decide we want to have the determinestic or random exploration type for training
# # So this line actually make trainig loop to be stochastic or deterministic
# # Cause I couldn't change the "collector = SyncDataCollector(... "  so I change the training loop.
# # By deafult and in original version it was stochastic.
# from torchrl.envs.utils import ExplorationType
# ELE_PPO_TRAINING_COLLECTOR_DETERMINISTIC = False  # set False for stochastic training

# if ELE_PPO_TRAINING_COLLECTOR_DETERMINISTIC:
#     ELE_PPO_TRAIN_EXPLORE_TYPE = ExplorationType.DETERMINISTIC
# else:
#     ELE_PPO_TRAIN_EXPLORE_TYPE = ExplorationType.RANDOM



# #  ELE_PPO_EVAL_EXPLORE_TYPE used just for evaluation part in training part, not in ele_exp_actor.py.
# # So:
# # During periodic evaluation inside ele_ppo_training.py : exploration type is whatever we set in ELE_PPO_EVAL_EXPLORE_TYPE. 
# # cause we leave it as DETERMINISTIC, our training is still probabilistic, but the logged “eval reward” is for the greedy policy.
# # If DETERMINISTIC : we’re measuring the performance of the greedy policy (argmax over action distribution).
# # If RANDOM : we’re measuring performance of the stochastic policy (sampling from categorical).
# # Cause in training we train nn and st(not ft cause we get ft in ele_exp_actor.py) I consider ELE_PPO_EVAL_EXPLORE_TYPE = ExplorationType.RANDOM, but again it doesn't affect training.
# # The only part that it affects is the logged “eval reward” during training which is used later on in plotting learning curves. However, since we compare "reward" not "eval reward" in learning curves it is fine if we even set it to deterministic.
# ELE_PPO_EVAL_EXPLORE_TYPE = ExplorationType.RANDOM #ExplorationType.DETERMINISTIC

# # endregion ==============================================================


# # region: constants for ele_exp_actor.py ====================================
# # ELE_ACTOR_VERSION = '20250505-192030'   #David's model
# # ELE_ACTOR_VERSION = '20250910-202015' # my model with 5 horizon
# # ELE_ACTOR_VERSION = '20250917-102249' # my model with 75 horizon
# # ELE_ACTOR_VERSION = '20250924-173308' # my model with 1 horizon with dirichlet alpha 0.5
# # ELE_ACTOR_VERSION = '20250924-183258' # my model with 1 horizon with reset prob [1,0,0,0,0]
# # ELE_ACTOR_VERSION = '20250924-184355' # my model with 5 horizon with reset prob [1,0,0,0,0]
# # ELE_ACTOR_VERSION = '20250924-190413' # my model with 10 horizon with reset prob [1,0,0,0,0]
# # ELE_ACTOR_VERSION = '20250925-100427'   # my model with 1 horizon with reset prob [1,0,0,0,0]
# # ELE_ACTOR_VERSION = '20250925-101620'   # my model with 5 horizon with reset prob [1,0,0,0,0]
# # ELE_ACTOR_VERSION = '20250930-152609'   # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# # ELE_ACTOR_VERSION = '20250930-163141_nn'   # my model with 5 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20250930-170010_st'   # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# # ELE_ACTOR_VERSION = '20251001-051210_nn'  # my model with 1 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251001-052254_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# # ELE_ACTOR_VERSION = '20251001-062631_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# # ELE_ACTOR_VERSION = '20251001-093701_nn'  # my model with 1 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251001-095252_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# # ELE_ACTOR_VERSION = '20251001-103449_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 6 and beta 1.0
# # ELE_ACTOR_VERSION = '20251001-110432_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0
# # ELE_ACTOR_VERSION = '20251001-112507_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 10 and beta 1.0


# # important files are:
# # ELE_ACTOR_VERSION = '20251001-134624_nn' # my model with 1 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251001-135623_st' # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0
# # ELE_ACTOR_VERSION = '20251001-141105_nn' # my model with 5 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251001-142834_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0
# # ELE_ACTOR_VERSION = '20251001-150733_nn' # my model with 10 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251001-153504_st' # my model with 10 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0

# # ELE_ACTOR_VERSION = '20251024-074359_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 9 and beta 1.0
# # ELE_ACTOR_VERSION = '20251025-180249_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 20.0


# # new file after regularization and frozen and change actor(name of actor model: before:actor_net for all) and training and chage name of saving files
# # ELE_ACTOR_VERSION = '20251112-191259_nn' # my model with 5 horizon with reset prob [1,0,0,0,0] - nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251112-194712_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 20.0 and regularization l1 1e-4
# # ELE_ACTOR_VERSION = '20251114-081217_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 2.0 and regularization none
# # ELE_ACTOR_VERSION = '20251114-091346_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0 and regularization none
# # ELE_ACTOR_VERSION = '20251114-102825_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 3 and beta 1.0 and regularization none
# # ELE_ACTOR_VERSION = '20251114-125549_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0 and regularization none
# # ELE_ACTOR_VERSION = '20251114-150609_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0 and regularization none
# # ELE_ACTOR_VERSION = '20251115-134927_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0 and commented regularization
# # ELE_ACTOR_VERSION = '20251115-150252_nn' # my model with 5 horizon with reset prob [1,0,0,0,0] - nn tree with 2 layers and 32 cells and commented regularization


# # ELE_ACTOR_VERSION = '20251115-171002_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0 and commented regularization chagned the code 
# # ELE_ACTOR_VERSION = '20251115-182749_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0 and commented regularization chagned the code 

# # ELE_ACTOR_VERSION = '20251115-185629_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0 and commented regularization chagned the code and corect sub_batch_size in training
# # ELE_ACTOR_VERSION = '20251117-061209_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0 corect sub_batch_size in training and back to the right code

# # ELE_ACTOR_VERSION = '20251117-132704_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0 corect sub_batch_size in training and back to the right code


# # Final models after all the corrections
# # ELE_ACTOR_VERSION = '20251118-131426_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 4 and beta 1.0 
# # ELE_ACTOR_VERSION = '20251118-161528_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0


# # The last files for comparisatin between nn, soft tree, and frozen tree
# # ELE_ACTOR_VERSION = '20251120-140227_nn' # my model with 5 horizon with reset prob [1,0,0,0,0] - nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251120-141843_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 5.0 non regularizatoin
# # ELE_ACTOR_VERSION = '20251120-145316_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0 non regularizatoin
# # ELE_ACTOR_VERSION = '20251120-151612_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 20.0 non regularizatoin

# # effect of regularization at beta = 20
# # ELE_ACTOR_VERSION = '20251120-174252_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 20.0, l1 regularizatoin, ELE_PPO_REG_LAMBDA=1
# # ELE_ACTOR_VERSION = '20251120-184216_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 20.0, l1 regularizatoin, ELE_PPO_REG_LAMBDA=1e-4
# # ELE_ACTOR_VERSION = '20251120-191201_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 20.0, l1 regularizatoin, ELE_PPO_REG_LAMBDA=1e-8


# # effect of regularization at beta = 2
# # ELE_ACTOR_VERSION = '20251120-193511_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 2.0, l1 regularizatoin, ELE_PPO_REG_LAMBD=1e-8

# # effect of regularization at beta = 10/9
# # ELE_ACTOR_VERSION = '20251120-201331_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 10/9, l1 regularizatoin, ELE_PPO_REG_LAMBDA=1e-8



# # effect of regularization at beta = 10/7
# # ELE_ACTOR_VERSION = '20251120-205327_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 10/7, l1 regularizatoin, ELE_PPO_REG_LAMBDA=1e-8
# # ELE_ACTOR_VERSION = '20251121-060644_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 10/7, l1 regularizatoin, ELE_PPO_REG_LAMBDA=1e-12


# # Switch to 4 condition states
# # ELE_ACTOR_VERSION = '20251124-184327_st' # my model with 5 horizon with reset prob [1,0,0,0] - soft tree with depth 8 and beta 10/7, l1 regularizatoin, ELE_PPO_REG_LAMBDA=1e-12
# # ELE_ACTOR_VERSION = '20251126-080810_nn' # nn tree with 2 layers and 32 cells




# # Switch to 4 condition state with cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.0]) not cs_pfs = stats.norm.cdf([-4.0, -3.5, -3.0, -2.5])
# # ELE_ACTOR_VERSION = '20251203-142137_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251203-144025_st' # 5 horizon, depth 8 and beta 1, none regularizatoin



# # I improve the actor and plot files to get the depth, beta, reg_type, reg_lambda from the init_params.npz file so no need to define them here again
# # so the previous asset versions will not work because they do not have these parameters saved in init_params.npz file
# # the following results is for cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])
# # ELE_ACTOR_VERSION = '20251203-162605_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251203-164941_st' # 5 horizon, depth 8 and beta 1, none regularizatoin


# # reset for cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.25])
# # ELE_ACTOR_VERSION = '20251203-173416_nn' # nn tree with 2 layers and 32 cells





# # reset for cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])  
# # ELE_ACTOR_VERSION = '20251203-180055_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251203-181754_st' # 5 horizon, depth 8 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION = '20251203-184753_st' # 5 horizon, depth 4 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION = '20251203-190649_st' # 5 horizon, depth 6 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION = '20251203-192359_st' # 5 horizon, depth 7 and beta 1, none regularizatoin


# # Change tempeture for cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5]) 
# # ELE_ACTOR_VERSION = '20251204-140355_st' # 5 horizon, depth 7, beta=5(T=0.2) , none regularizatoin
# # ELE_ACTOR_VERSION = '20251204-143654_st' # 5 horizon, depth 7, beta=2(T=0.5) , none regularizatoin
# # ELE_ACTOR_VERSION = '20251204-145953_st' # 5 horizon, depth 7, beta=0.2(T=5) , none regularizatoin
# # ELE_ACTOR_VERSION = '20251204-152746_st' # 5 horizon, depth 7, beta=10.0(T=0.1) , none regularizatoin
# # ELE_ACTOR_VERSION = '20251204-155307_st' # 5 horizon, depth 7, beta=20.0(T=0.05) , none regularizatoin



# # ELE_ACTOR_VERSION = '20251204-170339_st' # 5 horizon, depth 7, beta=1.5 , none regularizatoin
# # ELE_ACTOR_VERSION = '20251204-172616_st' # 5 horizon, depth 7, beta=1.75 , none regularizatoin



# # Depth =7
# # Beta = 1.75
# # Alpha value that I got from Drichlet = [0.28965147, 0.07418968, 0.04705171, 0.04048269]
# # Horizon = 20
# # Episode =100
# # ELE_ACTOR_VERSION = '20251205-083913_nn' # nn tree with 2 layers and 32 cells and dirichlet alpha from previous experiments and horizon 20
# # ELE_ACTOR_VERSION = '20251205-092637_st' #depth 7, beta=1.75 , none regularizatoin and dirichlet alpha from previous experiments and horizon 20





# # To verify the result, every thing is determinestic, Depth = 7 Beta = 1.75 Horizon = 5 episode= 1 everything is deterministic inital state = [1.0, 0.0, 0.0, 0.0]
# # ELE_ACTOR_VERSION ='20251211-152453_nn' # nn tree with 2 layers and 32 cells, determinestic
# # ELE_ACTOR_VERSION ='20251211-164428_st' # soft tree with  depth and 1.75 beta, determinestic

# # I switch back to peobabilistic
# # ELE_ACTOR_VERSION ='20251211-194543_nn'  # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION ='20251211-203053_st' # soft tree with  depth and 1000 beta, d=7, peobabilistic
# # ELE_ACTOR_VERSION ='20251212-042901_st' # soft tree with  depth and 1 beta, d=7, peobabilistic
# # ELE_ACTOR_VERSION ='20251212-045059_st' # soft tree with  depth and 20 beta, d=7, peobabilistic







# # I again consider the following assumption to verify find the reason the difference between average reward in training and evaluation
# # Depth =7
# # Beta = 1.75
# # Alpha value that I got from Drichlet = [0.28965147, 0.07418968, 0.04705171, 0.04048269]
# # Horizon = 20
# # Episode =100
# # cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])  for 4 cs
# # ELE_PPO_MAX_COST = 1 
# # ELE_PPO_EVAL_EXPLORE_TYPE = ExplorationType.RANDOM
# # ELE_PPO_TRAIN_EXPLORE_TYPE = ExplorationType.RANDOM
# # ELE_ACTOR_EXPLORE_TYPE_NN = ExplorationType.DETERMINISTIC 
# # ELE_ACTOR_EXPLORE_TYPE_ST = ExplorationType.DETERMINISTIC 
# # ELE_ACTOR_VERSION ='20251215-191317_nn'  # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION ='20251216-053246_st' # soft tree with  depth and 1.75 beta, d=7, peobabilistic training and deterministic evaluation




# # folloiwng results are for girder beam with 4 condition states(AASHTO) for all states and new alpha for dirichlet with cfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])
# # ELE_ACTOR_VERSION = '20251217-150810_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION = '20251217-164927_st' # soft tree with  depth and 1.75 beta, d=7, peobabilistic training and deterministic evaluation
# # ELE_ACTOR_VERSION = '20251217-171254_st' # soft tree with  depth and 1.75 beta, d=8, peobabilistic training and deterministic evaluation
# # ELE_ACTOR_VERSION = '20251217-174134_st' # soft tree with  depth and 1 beta, d=8, peobabilistic training and deterministic evaluation
# # ELE_ACTOR_VERSION = '20251218-182531_st' # soft tree with  depth and 10000 beta, d=8, peobabilistic training and deterministic evaluation
# # ELE_ACTOR_VERSION = '20251218-184848_st' # soft tree with  depth and 10 beta, d=8, peobabilistic training and deterministic evaluation
# # ELE_ACTOR_VERSION = '20251218-195034_st' # soft tree with  depth and 100 beta, d=8, peobabilistic training and deterministic evaluation
# # ELE_ACTOR_VERSION = '20251219-035017_st' # soft tree with  depth and 1000 beta, d=8, peobabilistic training and deterministic evaluation





# # folloiwng results are for girder beam with 4 condition states(AASHTO) for oregon and new alpha for dirichlet with cfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])
# # ELE_ACTOR_VERSION = '20251222-132849_nn' # nn tree with 2 layers and 32 cells
# ELE_ACTOR_VERSION = '20251222-134720_st' # soft tree with  depth and 1 beta, d=8, peobabilistic training and deterministic evaluation








# # If 'nn': evaluate the trained neural network actor
# # If 'st': evaluate the trained soft decision tree actor
# # If 'ft': build a frozen oblique tree from the soft tree and evaluate that
# ELE_PPO_EVAL_ACTOR = 'st'  # 'nn', 'st', 'ft'

# # if ELE_PPO_EVAL_ACTOR == 'st' or ELE_PPO_EVAL_ACTOR == 'ft':
# #     # soft tree parameters
# #     depth_soft_actor = 8 #8 #5
# #     beta_soft_actor = 1 #10/7 #20.0
# #     batchnorm_soft_actor = False

# #     # Frozen-tree pruning and μ-estimation knobs
# #     # Threshold: if for an internal node all |w_j|<threshold (j>=1, ignoring bias),
# #     # we prune that node and replace it by a synthesized leaf.
# #     ELE_PPO_REG_TYPE_ACTOR = "none" #"l1"     # options: "none", "l1", "l2", "groupl1"

# #     if ELE_PPO_REG_TYPE_ACTOR != "none":
# #         ELE_PPO_REG_LAMBDA_ACTOR = 1e-12 # regularization strength


# if ELE_PPO_EVAL_ACTOR == 'st' or ELE_PPO_EVAL_ACTOR == 'ft':
#     # Freeze threshold selection
#     if ELE_PPO_REG_TYPE == "none":
#         ELE_PPO_FREEZE_THRESHOLD_ACTOR = -np.inf  #0.0 means no pruning
#     elif ELE_PPO_REG_TYPE == "l1" or ELE_PPO_REG_TYPE == "l2" or ELE_PPO_REG_TYPE == "groupl1":
#         ELE_PPO_FREEZE_THRESHOLD_ACTOR = 0.5 #2



# # elif ELE_PPO_EVAL_ACTOR == 'nn':
# #     ELE_PPO_REG_TYPE_ACTOR = "none" # I need to define this to avoid error in ele_ppo_training.py(line: import test_constants)




# # If 0: use the evaluation horizon already defined in ele_exp_actor.py
# # If >0: you can override how many steps you sample for μ estimation.
# ELE_PPO_FREEZE_MU_STEPS = 0

# # Whether to save the frozen wiring as frozen_tree.npz next to actor_net_state_dict.pt
# ELE_PPO_FREEZE_SAVE_WIRING = True


# # Optional class names (for nicer probability printing)
# # If we have 2 actions, for example:
# # class_names = ["Do nothing", "repair"] or we can set None
# class_names = ["DN", "Main", "Repair", "Rehab", "Rep"] #None  # or set your own list ["DN", "M", "R", "Reh", "T"]


# ELE_ACTOR_HORIZON = 5 #20 #20 #5 #20 #5 #75
# # ELE_ACTOR_N_HORIZON = 1
# ELE_ACTOR_N_EPISODES = 1 #100000 #100000 #100 #1 # modified to avoid confusion
# # Don't forger when you change the follwiong parameter we need to first run the ele_exp_actor.py and then run the plt_nn_st.py
# ELE_ACTOR_MAX_COST = 1 #unit_costs.max()

# ELE_ACTOR_RESET_PROB = None
# ELE_ACTOR_DIRICHLET_ALPHA = [0.05594704, 0.16108377, 0.05494736, 0.03863813] # For all states: [0.15481776, 0.07666929, 0.04912562, 0.03946825] # Alpha for girder beam # [0.28965147, 0.07418968, 0.04705171, 0.04048269]:alpha for three prestressed elements  # 0.5*np.ones(NCS)
# ELE_ACTOR_RANDOM_STATE = 42
# # ELE_ACTOR_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0]) #np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # ELE_ACTOR_RESET_PROB = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # ELE_ACTOR_RESET_PROB = np.array([0.0, 0.8, 0.2, 0.0, 0.0])
# # ELE_ACTOR_DIRICHLET_ALPHA = None
# # ELE_ACTOR_RANDOM_STATE = 'off'



# # So ELE_ACTOR_EXPLORE_TYPE controls:
# # How we sample actions when estimating μ (for freezing the soft tree), and
# # How we sample actions when we gather experience / action histograms in ele_exp_actor.py.
# # so I need to define two different exploration types:ELE_ACTOR_EXPLORE_TYPE_MU and ELE_ACTOR_EXPLORE_TYPE
# # What should mu-rollout use?
# # For mu estimation, we are using soft actor + ProbabilisticActor:
# # soft_prob_actor = ProbabilisticActor(... distribution_class=CategoricalDist, ...)
# # If we set exploration to DETERMINISTIC, we’re getting the mu under the greedy soft policy (argmax of logits).
# # If we set it to RANDOM, we’re getting mu under the true stochastic soft policy (sampling from the categorical defined by logits).
# # The frozen tree is built to approximate the original stochastic soft policy’s behavior. 
# # The frozen tree itself is deterministic later, but that doesn’t mean mu must be computed with deterministic argmax. The tree uses mu to know which branches matter, and that’s naturally defined under the stochastic policy.
# # Conclusion: To ensure reproducible μ estimation, the environment random seed is fixed so that the initial state sampled 
# # from the Dirichlet distribution is reproducible across runs. However, the states visited in future horizons depend 
# # not only on the initial state but also on the actions taken at each step. 
# # If actions are sampled randomly, different action sequences can lead to different state trajectories 
# # and therefore different μ values, even when the initial state is identical. For this reason, 
# # `ELE_ACTOR_EXPLORE_TYPE_MU` is set to DETERMINISTIC, ensuring that the same action is selected at each horizon given the same state. 
# # This makes the entire rollout trajectory, and consequently the estimated μ, fully reproducible. 
# # Although μ could also be made reproducible with `RANDOM` exploration 
# # by explicitly seeding all action-related random number generators, doing so would affect subsequent rollouts later 
# # in `ele_exp_actor.py`, so I forgert to set random seed. To avoid unintended side effects and maintain clean separation between μ estimation and evaluation, 
# # deterministic exploration is used for μ.
# ELE_ACTOR_EXPLORE_TYPE_MU = ExplorationType.DETERMINISTIC #ExplorationType.RANDOM 


# # Conceptually: nn and soft tree produce probabilities, but that does not mean we must sample from them at evaluation time.
# # Also when I actually freeze and deploy, I usually want a stable, deterministic maintenance policy (bridge engineer doesn’t want “today we randomly choose repair instead of maintenance”).
# # Later I though to define two different ELE_ACTOR_EXPLORE_TYPE for st/nn and ft. But for a fair comparison among nn,st and ft, I do not use “RANDOM for nn/st, DETERMINISTIC for ft”. Why?
# # Because If I set 
# # nn, st → ExplorationType.RANDOM
# # ft → ExplorationType.DETERMINISTIC
# # then nn/st policies will sometimes pick sub-optimal but non-zero-probability actions.
# # ft will never pick those sub-optimal actions (it’s deterministic).
# # So we are comparing “noisy” stochastic policies vs a “clean” deterministic one.
# # That usually artificially lowers the observed performance of nn/st relative to ft. 
# # It’s like comparing “student taking the exam sober” vs “student taking the exam while periodically rolling a die to change answers.”
# ELE_ACTOR_EXPLORE_TYPE_NN = ExplorationType.RANDOM # ExplorationType.DETERMINISTIC 
# ELE_ACTOR_EXPLORE_TYPE_ST = ExplorationType.RANDOM # ExplorationType.DETERMINISTIC 
# ELE_ACTOR_EXPLORE_TYPE_FT = ExplorationType.DETERMINISTIC # This must be deterministic to choose greedy action because the frozen tree chooses the action with max prob
# # endregion ==============================================================


# region: constants for DPvsPPO.py ==================================
ELE_DP_HORIZON = 5 #75
ELE_DP_N_EPISODES = 10000 # In DP we always consider 1 episode
ELE_DP_MAX_COST = 1.0

ELE_DP_INC_STEP = True
# if ELE_DP_INC_STEP:
#     ELE_DP_INPUT_DIM = NCS + 1
# else:
#     ELE_DP_INPUT_DIM = NCS

ELE_DP_RESET_PROB = None
ELE_DP_DIRICHLET_ALPHA = [0.14964171, 0.11136174, 0.05003725, 0.03926025] #[0.15481776, 0.07666929, 0.04912562, 0.03946825] # Alpha for girder beam #0.5*np.ones(NCS)
ELE_DP_RANDOM_STATE = 42
# ELE_DP_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0]) #np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # ELE_DP_RESET_PROB = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # ELE_DP_RESET_PROB = np.array([0.0, 0.8, 0.2, 0.0, 0.0])
# ELE_DP_DIRICHLET_ALPHA = None
# ELE_DP_RANDOM_STATE = 'off'

ELE_DP_EXPLORE_TYPE = ExplorationType.DETERMINISTIC # This must be deterministic to choose greedy action because DP policy is deterministic

# # endregion ================================================================


# # region: constants for pygad_reliability.py ==================================
# ELE_GA_SEED_FOR_PyGAD = 0
# ELE_GA_POP = 128                            #benchmark = 128       # Population size - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)
# ELE_GA_GENS = 256                           #benchmark = 256       # Number of generations - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)

# ELE_GA_LB_BETA = norm.ppf(1-max(cs_pfs))  # 2.0
# ELE_GA_UB_BETA = norm.ppf(1-min(cs_pfs))  # 4.2

# ELE_GA_KEEP_PARENTS = 13                     #benchmark = 13       # 10% of pop=128 / number of parents to keep in the next generation  

# ELE_GA_PARENT_SELECTION = "tournament"       # benchmark:"tournament"
# K_TOURNAMENT=3                               # benchmark = 3

# ELE_GA_CROSSOVER_TYPE = "uniform"            # benchmark = "uniform"     Rationale: cause  genes are continuous β-thresholds; Randomly selects each gene from one of the parents


# ELE_GA_CROSSOVER_PROBABILITY = None
# if ELE_GA_CROSSOVER_TYPE == "uniform" or ELE_GA_CROSSOVER_TYPE == "scattered":
#     """
#     - PyGAD compares each gene in the two parent solutions.
#         - For each gene 'position':
#         - It generates a random number between 0 and 1.
#             - If that number is less than 0.7, the gene is swapped between the parents.
#             - If it's greater than or equal to 0.7, the gene is kept as-is.
#     """
#     ELE_GA_CROSSOVER_PROBABILITY = None      # benchmark =(None means crossover is applied to every mating pair)  - This parameter is used only in 'uniform' crossover or scattered crossover



# MUTATION_TYPE="random"                    # benchmark = random # mutate genes by drawing random numbers (as opposed to e.g. swap/scramble for permutations). Best for continuous genes like your β-thresholds.
#                                           # genes by either replacing them or nudging them — depending on the value of mutation_by_replacement
# MUTATION_BY_REPLACEMENT=False             # benchmark = False    # nudge instead of replace / genes by either replacing them(True) or nudging them(False)
# RANDOM_MUTATION_MIN_VAL=-1.0 #-0.10       # benchmark = -1.0
# RANDOM_MUTATION_MAX_VAL=1.0 #+0.10        # benchmark = 1.0   # small β step (+0.10)
# MUTATION_NUM_GENES = 1                    # benchmark =1 / 1 number of genes to mutate in a solution


# ELE_GA_MAX_COST = unit_costs.max()

# # Initial distribution control (reset-style)
# # ELE_GA_RESET_PROB = None
# # ELE_GA_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# # ELE_GA_RANDOM_STATE = 42
# ELE_GA_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # ELE_GA_RESET_PROB = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # ELE_GA_RESET_PROB = np.array([0.0, 0.0, 0.1, 0.8, 0.1])

# ELE_GA_DIRICHLET_ALPHA = None
# ELE_GA_RANDOM_STATE = 'off'

# # Inputs for Evaluation part: To compare GA with PPO(evaluation part)
# ELE_GA_HORIZON = 5 #35
# ELE_GA_N_EPISODES_EVAL = 1 # modified to avoid confusion
# ELE_GA_MAX_COST_EVAL = 1.0

# # ELE_GA_RESET_PROB_EVAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # ELE_GA_DIRICHLET_ALPHA_EVAL = None
# # ELE_GA_RANDOM_STATE_EVAL = 'off'
# ELE_GA_RESET_PROB_EVAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # ELE_GA_RESET_PROB_EVAL = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # ELE_GA_RESET_PROB_EVAL = np.array([0.0, 0.0, 0.1, 0.8, 0.1])

# ELE_GA_DIRICHLET_ALPHA_EVAL = None
# ELE_GA_RANDOM_STATE_EVAL = 'off'

# ELE_GA_INC_STEP_EVAL = True

# ELE_GA_EXPLORE_TYPE_EVAL = ExplorationType.DETERMINISTIC



# # ELE_GA_SEED_FOR_PyGAD = 0
# # ELE_GA_POP = 512 #512#128 #80                        # Population size - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)
# # ELE_GA_GENS = 1024# 1500 #256 #256 #200               # Number of generations - (Population * Genes = 128*256) ~ (PPO frames/horizon = 5*32*1024/5 = 32768)
# # # ELE_GA_LB_BETA, ELE_GA_UB_BETA = 0.0, 8.0          # typical β range (pf ~ 0.5 down to 1e-15)
# # ELE_GA_LB_BETA = norm.ppf(1-max(cs_pfs))  # 2.0
# # ELE_GA_UB_BETA = norm.ppf(1-min(cs_pfs))  # 4.2

# # ELE_GA_KEEP_PARENTS = 2                             # 10% of pop=128 / number of parents to keep in the next generation  

# # # ELE_GA_PARENT_SELECTION = "sss"                    # steady-state selection
# # ELE_GA_PARENT_SELECTION = "tournament"               # deafult:parent_selection_type="sss"
# # K_TOURNAMENT=20 #5 # deafult = 3

# # # ELE_GA_CROSSOVER_TYPE = "single_point"             # single point means Only one point is used to split and recombine the genes(randolyn selected)
# # ELE_GA_CROSSOVER_TYPE = "uniform"                    #deafult = crossover_type="single_point"     Rationale: cause  genes are continuous β-thresholds; Randomly selects each gene from one of the parents


# # ELE_GA_CROSSOVER_PROBABILITY = None
# # if ELE_GA_CROSSOVER_TYPE == "uniform" or ELE_GA_CROSSOVER_TYPE == "scattered":
# #     """
# #     - PyGAD compares each gene in the two parent solutions.
# #         - For each gene 'position':
# #         - It generates a random number between 0 and 1.
# #             - If that number is less than 0.7, the gene is swapped between the parents.
# #             - If it's greater than or equal to 0.7, the gene is kept as-is.
# #     """
# #     ELE_GA_CROSSOVER_PROBABILITY = None       #default = None(None means crossover is applied to every mating pair)  - This parameter is used only in 'uniform' crossover or scattered crossover



# # MUTATION_TYPE="random"                    # mutate genes by drawing random numbers (as opposed to e.g. swap/scramble for permutations). Best for continuous genes like your β-thresholds.
# #                                           # genes by either replacing them or nudging them — depending on the value of mutation_by_replacement
# # MUTATION_BY_REPLACEMENT=False             # deafult = False    # nudge instead of replace / genes by either replacing them(True) or nudging them(False)
# # RANDOM_MUTATION_MIN_VAL=-0.10       # deafult = -1.0
# # RANDOM_MUTATION_MAX_VAL=0.10        # deafult = 1.0   # small β step (+0.10)
# # MUTATION_NUM_GENES = 3                    # 1 number of genes to mutate in a solution



# # ELE_GA_MAX_COST = unit_costs.max()

# # # Initial distribution control (reset-style)
# # # ELE_GA_RESET_PROB = None
# # # ELE_GA_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# # # ELE_GA_RANDOM_STATE = 42
# # ELE_GA_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # # ELE_GA_RESET_PROB = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # # ELE_GA_RESET_PROB = np.array([0.0, 0.0, 0.1, 0.8, 0.1])

# # ELE_GA_DIRICHLET_ALPHA = None
# # ELE_GA_RANDOM_STATE = 'off'

# # # Inputs for Evaluation part: To compare GA with PPO(evaluation part)
# # ELE_GA_HORIZON = 5 #35
# # ELE_GA_N_EPISODES_EVAL = 1 # modified to avoid confusion
# # ELE_GA_MAX_COST_EVAL = 1.0

# # # ELE_GA_RESET_PROB_EVAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # # ELE_GA_DIRICHLET_ALPHA_EVAL = None
# # # ELE_GA_RANDOM_STATE_EVAL = 'off'
# # ELE_GA_RESET_PROB_EVAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# # # ELE_GA_RESET_PROB_EVAL = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# # # ELE_GA_RESET_PROB_EVAL = np.array([0.0, 0.0, 0.1, 0.8, 0.1])

# # ELE_GA_DIRICHLET_ALPHA_EVAL = None
# # ELE_GA_RANDOM_STATE_EVAL = 'off'

# # ELE_GA_INC_STEP_EVAL = True

# # ELE_GA_EXPLORE_TYPE_EVAL = ExplorationType.DETERMINISTIC
# # # endregion ==============================================================


# # region: which actor model compared(leaning curve) for Plt_LC_nn_st.py ==================================
# # reg_type_plot = "non" #"l1"
# # reg_lambda_plot = "0" #1e-12

# # nn with 5 horizon and st with 5 horizon and depth=8 and beta=1.0
# # ELE_ACTOR_VERSION_nn = '20251001-141105_nn'
# # ELE_ACTOR_VERSION_st = '20251001-142834_st'

# # # nn with 5 horizon and st with 5 horizon and depth=9 and beta=1.0
# # ELE_ACTOR_VERSION_nn = '20251001-141105_nn'
# # ELE_ACTOR_VERSION_st = '20251024-074359_st'


# # # nn with 5 horizon and st with 5 horizon and depth=8 and beta=20.0
# # ELE_ACTOR_VERSION_nn = '20251001-141105_nn'
# # ELE_ACTOR_VERSION_st = '20251025-180249_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 20.0

# # After corrections I got the following files: 
# # nn with 1 horizon and st with 1 horizon and depth=8
# # ELE_ACTOR_VERSION_nn = '20251120-140227_nn'
# # ELE_ACTOR_VERSION_st = '20251120-141843_st' # beta 5.0 non regularizatoin
# # ELE_ACTOR_VERSION_st = '20251120-145316_st' # beta 1.0 non regularizatoin
# # ELE_ACTOR_VERSION_st = '20251120-151612_st' # beta 20.0 non regularizatoin
# # ELE_ACTOR_VERSION_st = '20251121-060644_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 10/7, l1 regularizatoin, ELE_PPO_REG_LAMBDA=1e-12


# # Switch to 4 condition state
# # ELE_ACTOR_VERSION_nn = '20251126-080810_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION_st = '20251124-184327_st' # my model with 5 horizon with reset prob [1,0,0,0] - soft tree with depth 8 and beta 10/7, l1 regularizatoin, ELE_PPO_REG_LAMBDA=1e-12


# # fine tuning for the 4-cs env
# # ELE_ACTOR_VERSION_nn = '20251203-142137_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION_st = '20251203-144025_st' # 5 horizon, depth 8 and beta 1, none regularizatoin



# # The previous versions for 5-cs env don't work here because the init_params.npz file does not have the correct parameters for 4-cs env
# # ELE_ACTOR_VERSION_nn = '20251203-162605_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION_st = '20251203-164941_st' # 5 horizon, depth 8 and beta 1, none regularizatoin


# # reset for cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.25])
# # ELE_ACTOR_VERSION_nn = '20251203-173416_nn' # nn tree with 2 layers and 32 cells


# # reset for cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])
# # ELE_ACTOR_VERSION_nn = '20251203-180055_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION_st = '20251203-181754_st' # 5 horizon, depth 8 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION_st = '20251203-184753_st' # 5 horizon, depth 4 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION_st = '20251203-190649_st' # 5 horizon, depth 6 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION_st = '20251203-192359_st' # 5 horizon, depth 7 and beta 1, none regularizatoin


# # Change tempeture for cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5]) for depth=7
# # ELE_ACTOR_VERSION_st = '20251204-140355_st' # 5 horizon, depth 7, beta=5(T=0.2) , none regularizatoin
# # ELE_ACTOR_VERSION_st = '20251204-143654_st' # 5 horizon, depth 7, beta=2(T=0.5) , none regularizatoin
# # ELE_ACTOR_VERSION_st = '20251204-145953_st' # 5 horizon, depth 7, beta=0.2(T=5) , none regularizatoin
# # ELE_ACTOR_VERSION_st = '20251204-152746_st' # 5 horizon, depth 7, beta=10.0(T=0.1) , none regularizatoin
# # ELE_ACTOR_VERSION_st = '20251204-155307_st' # 5 horizon, depth 7, beta=20.0(T=0.05) , none regularizatoin
# # ELE_ACTOR_VERSION_st = '20251204-172616_st' # 5 horizon, depth 7, beta=1.75 , none regularizatoin



# # ELE_ACTOR_VERSION_nn = '20251205-083913_nn' # nn tree with 2 layers and 32 cells and dirichlet alpha from previous experiments and horizon 20
# # ELE_ACTOR_VERSION_st = '20251205-092637_st' #depth 7, beta=1.75 , none regularizatoin and dirichlet alpha from previous experiments and horizon 20


# # ELE_ACTOR_VERSION_nn ='20251211-152453_nn' # nn tree with 2 layers and 32 cells every thing is deterministic
# # ELE_ACTOR_VERSION_st ='20251211-164428_st' # soft tree with 7 depth and 1.75 beta every thing is deterministic  

# # ELE_ACTOR_VERSION_nn ='20251211-194543_nn'  # nn tree with 2 layers and 32 cells , probabilistic
# # ELE_ACTOR_VERSION_st ='20251211-203053_st' # soft tree with  depth and 1000 beta, d=7 , probabilistic
# # ELE_ACTOR_VERSION_st ='20251212-042901_st' # soft tree with  depth and 1 beta, d=7 , probabilistic
# # ELE_ACTOR_VERSION_st ='20251212-045059_st' # soft tree with  depth and 20 beta, d=7 , probabilistic





# # I again consider the following assumption to verify find the reason the difference between average reward in training and evaluation
# # Depth =7
# # Beta = 1.75
# # Alpha value that I got from Drichlet = [0.28965147, 0.07418968, 0.04705171, 0.04048269]
# # Horizon = 20
# # Episode =100
# # cs_pfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])  for 4 cs
# # ELE_PPO_MAX_COST = 1 
# # ELE_PPO_EVAL_EXPLORE_TYPE = ExplorationType.RANDOM
# # ELE_PPO_TRAIN_EXPLORE_TYPE = ExplorationType.RANDOM
# # ELE_ACTOR_EXPLORE_TYPE_NN = ExplorationType.DETERMINISTIC 
# # ELE_ACTOR_EXPLORE_TYPE_ST = ExplorationType.DETERMINISTIC 
# # ELE_ACTOR_VERSION_nn ='20251215-191317_nn'  # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION_st ='20251216-053246_st' # soft tree with  depth and 1.75 beta, d=7, peobabilistic training and deterministic evaluation




# # folloiwng results are for girder beam with 4 condition states(AASHTO) and new alpha for dirichlet with cfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])
# ELE_ACTOR_VERSION_nn = '20251217-150810_nn' # nn tree with 2 layers and 32 cells
# ELE_ACTOR_VERSION_st = '20251217-174134_st' # soft tree with  depth and 1 beta, d=8, peobabilistic training and deterministic evaluation



# WINDOW = 50  # for rolling average - integer

# # endregion ==============================================================

# # region: which actor model compared(leaning curve) for Plt_LC_nn_st.py for different tempeture and different depth ==================================
# # # Effect of depth for soft tree with 5 horizon, beta=1.0, no regularization
# # ELE_ACTOR_VERSION_nn_vs_st = '20251203-180055_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION_st_1 = '20251203-184753_st' # 5 horizon, depth 4 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION_st_2 = '20251203-190649_st' # 5 horizon, depth 6 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION_st_3 = '20251203-192359_st' # 5 horizon, depth 7 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION_st_4 = '20251203-181754_st' # 5 horizon, depth 8 and beta 1, none regularizatoin
# # actor_st_versions = [ELE_ACTOR_VERSION_st_1, ELE_ACTOR_VERSION_st_2, ELE_ACTOR_VERSION_st_3, ELE_ACTOR_VERSION_st_4]



# # ELE_ACTOR_VERSION_nn_vs_st = '20251203-180055_nn' # nn tree with 2 layers and 32 cells
# # ELE_ACTOR_VERSION_st_1 = '20251204-155307_st' # 5 horizon, depth 7, beta=20.0(T=0.05) , none regularizatoin
# # ELE_ACTOR_VERSION_st_2 = '20251204-152746_st' # 5 horizon, depth 7, beta=10.0(T=0.1) , none regularizatoin
# # # ELE_ACTOR_VERSION_st_3 = '20251204-140355_st' # 5 horizon, depth 7, beta=5(T=0.2), none regularizatoin
# # # ELE_ACTOR_VERSION_st_4 = '20251204-143654_st' # 5 horizon, depth 7, beta=2(T=0.5) , none regularizatoin
# # ELE_ACTOR_VERSION_st_5 = '20251203-192359_st' # 5 horizon, depth 7 and beta 1, none regularizatoin
# # ELE_ACTOR_VERSION_st_6 = '20251204-145953_st' # 5 horizon, depth 7, beta=0.2(T=5), none regularizatoin

# # # actor_st_versions = [ELE_ACTOR_VERSION_st_1, ELE_ACTOR_VERSION_st_2, ELE_ACTOR_VERSION_st_3, ELE_ACTOR_VERSION_st_4, ELE_ACTOR_VERSION_st_5, ELE_ACTOR_VERSION_st_6]
# # actor_st_versions = [ELE_ACTOR_VERSION_st_1, ELE_ACTOR_VERSION_st_2, ELE_ACTOR_VERSION_st_5 ,ELE_ACTOR_VERSION_st_6]


# # ELE_ACTOR_VERSION_nn_vs_st ='20251211-194543_nn'  # nn tree with 2 layers and 32 cells , probabilistic
# # ELE_ACTOR_VERSION_st_1 ='20251212-042901_st' # soft tree with  depth and 1 beta, d=7 , probabilistic
# # ELE_ACTOR_VERSION_st_2 ='20251212-045059_st' # soft tree with  depth and 20 beta, d=7 , probabilistic
# # ELE_ACTOR_VERSION_st_3 ='20251211-203053_st' # soft tree with  depth and 1000 beta, d=7 , probabilistic
# # actor_st_versions = [ELE_ACTOR_VERSION_st_1, ELE_ACTOR_VERSION_st_2, ELE_ACTOR_VERSION_st_3]





# # folloiwng results are for girder beam with 4 condition states(AASHTO) and new alpha for dirichlet with cfs = stats.norm.cdf([-4.2, -3.5, -3.0, -2.5])
# ELE_ACTOR_VERSION_nn_vs_st = '20251217-150810_nn' # nn tree with 2 layers and 32 cells
# ELE_ACTOR_VERSION_st_1 = '20251217-174134_st' # soft tree with  depth and 1 beta, d=8, peobabilistic training and deterministic evaluation
# # ELE_ACTOR_VERSION_st_2 = '20251218-184848_st' # soft tree with  depth and 10 beta, d=8, peobabilistic training and deterministic evaluation
# # ELE_ACTOR_VERSION_st_3 = '20251218-195034_st' # soft tree with  depth and 100 beta, d=8, peobabilistic training and deterministic evaluation
# # ELE_ACTOR_VERSION_st_4 = '20251219-035017_st' # soft tree with  depth and 1000 beta, d=8, peobabilistic training and deterministic evaluation
# ELE_ACTOR_VERSION_st_5 = '20251218-182531_st' # soft tree with  depth and 10000 beta, d=8, peobabilistic training and deterministic evaluation
# actor_st_versions = [ELE_ACTOR_VERSION_st_1, ELE_ACTOR_VERSION_st_5]

# # endregion ==============================================================



