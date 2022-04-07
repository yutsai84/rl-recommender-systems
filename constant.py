ROOT_DIR = "C:/Users/jacob/Documents/GitHub/rl-recommender-systems/src"
ROOT_DIR = f"{ROOT_DIR}/artifacts"  # @param {type:"string"} Root directory for writing logs/summaries/checkpoints.
ARTIFACTS_DIR = f"{ROOT_DIR}/artifacts"  # @param {type:"string"} Where the trained model will be saved and restored.
PROFILER_DIR = f"{ROOT_DIR}/profiler"  # @param {type:"string"} Directory for TensorBoard Profiler artifacts.
DATA_PATH = f"{ROOT_DIR}/ml-100k/u.data" # @param {type:"string"}"gs://cloud-samples-data/vertex-ai/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/u.data"  # Location of the MovieLens 100K dataset's "u.data" file.

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