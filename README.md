# Reinforcement Learning (RL) Recommender Systems Pipeline using MovieLens 100K dataset

## TF-Agents
* A well written and detailed summary is [here](https://cloud.google.com/blog/topics/developers-practitioners/build-reinforcement-learning-recommendation-application-using-vertex-ai)
* A flexible library enabling one to implement and test RL-based applications
* Policy: Equivalent of a model in a supervised learning model, mapping from an observation to an action
* Action: output of a policy
* Agent: an algorithm that uses a policy to output an action, and trains the policy
* Observation: 
* Environment: At each time step, the environment generates an observation, and respond with a reward as feedback given actions taken by an agent and the perceived observation

## Seting up an experiment
* Users are represented as user embedding vectors
* Actions are the movies available for recommendation
* Reward is user rating of a movie

## Training a RL recommender system
* During training, randomly sample users, use those user vectors as observations to query some policy for items to recommend, and then apply that recommendation to users and obtain their feedback as rewards

## Making a prediction 
* Given an obervation, which is a user vector and trained policy, return predicted feedback, that is predicted rating for movies 

## Code Structure
* This code base leverages and follow closely with the step-by-step tutorials from a [google blog post](https://github.com/yutsai84/vertex-ai-samples/tree/master/community-content/tf_agents_bandits_movie_recommendation_with_kfp_and_vertex_sdk/step_by_step_sdk_tf_agents_bandits_movie_recommendation)
* Model parameters and i/o are defined in constants module
* Raw data of movie lens is expected to be downloaded from [kaggle](https://www.kaggle.com/prajitdatta/movielens-100k-dataset) and is saved to the path below defined in constants module

```
DATA_PATH = f"{ROOT_DIR}/ml-100k/u.data" 
```
* The source code is located at src folder, consisting of modeling, predicting, and plotting
* Jupyter notebook is used as a runner code