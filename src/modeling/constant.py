# Set hyperparameters.
BATCH_SIZE = 8  # @param {type:"integer"} Training and prediction batch size.
TRAINING_LOOPS = 5  # @param {type:"integer"} Number of training iterations.
STEPS_PER_LOOP = 2  # @param {type:"integer"} Number of driver steps per training iteration.

# Set MovieLens simulation environment parameters.
RANK_K = 20  # @param {type:"integer"} Rank for matrix factorization in the MovieLens environment; also the observation dimension.
NUM_ACTIONS = 20  # @param {type:"integer"} Number of actions (movie items) to choose from.
PER_ARM = False  # Use the non-per-arm version of the MovieLens environment.

# Set agent parameters.
TIKHONOV_WEIGHT = 0.001  # @param {type:"number"} LinUCB Tikhonov regularization weight.
AGENT_ALPHA = 10.0  # @param {type:"number"} LinUCB exploration parameter that multiplies the confidence intervals.