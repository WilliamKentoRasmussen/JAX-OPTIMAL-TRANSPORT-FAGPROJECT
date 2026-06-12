MODELS_DIM = [2,8, 16, 32]  # 128
INTERMEDIATE_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
GAMMA = [0.0001,0.001,0.01,0.1,1]  # regularization strength for Sinkhorn
MAX_POINTS = 1000  # amount of max decoded images
MAX_ITERATION = 1000  # Sinkhorn iterations
LABELS = [0,1,2,3,4,5,6,7,8,9]

TQDM_SINKHORN = True
VERBOSE_OPTIMAL_TRANSPORT = False
SAVE_INTERMEDIATE = False 
