# Reinforcement Learning using AI Platform

Reinforcement learning (RL) using Clould AI Platform gives you two key benefits:  
* You can train many models in parallel, this allows you to quickly iterate your
  concept and you only pay for the compute/storage resources you use in each
  job.
* You benefit from the managed hyper-parameter tuning service, which typically
  results in quicker convergence than a naive grid search.  

We present examples of running RL algorithms on AI Platform. Specifically, we provide implementations of [DDPG](https://arxiv.org/abs/1509.02971), [TD3](https://arxiv.org/abs/1802.09477) and C2A2 (a modification based on TD3), train and tune their hyper-parameters in the [BipedalWalker](https://gym.openai.com/envs/BipedalWalker-v2) environment.

Read more about running RL on AI Platform
[here](https://cloud.google.com/blog/products/ai-machine-learning/deep-reinforcement-learning-on-gcp-using-hyperparameters-and-cloud-ml-engine-to-best-openai-gym-games?fbclid=IwAR1i2Q-J_FXs8cZifkt7K8u5xWXzwM_U6Ls6KpMA0utVifhvsDTpLKkPGo4).  

## Run the code on GCP

We assume you have created a google cloud project and have a google cloud
storage bucket that has the project id as its name. You can find more about how to
create project and bucket [here](https://cloud.google.com/docs/).  

To train an agent with default parameters:

```shell
bash start.sh -p your-google-cloud-project-id
```

To tune hyper-parameters:

```shell
bash start.sh -p your-google-cloud-project-id -t
```

Check out start.sh for more options.

## Results

### Learning curves from hyper-parameter tuning
Learning curves from hyper-parameter tuning.  
![Learning curves](https://storage.googleapis.com/gcp_blog/img/learning_curve.png)  
Learning curve from the trial with best scores.  
BipedalWalker is considered “solved” if the agent can reach an average reward of 300 for 100 episodes in a row.
The agent we trained meet the criteria before episode #2000 and this can put us at the third place on the [leaderboard](https://github.com/openai/gym/wiki/Leaderboard).
![Optimal curve](https://storage.googleapis.com/gcp_blog/img/optimal_curve.png)  

### Learnt TD3 agent
![ep100](https://storage.googleapis.com/gcp_blog/img/ep100.gif)  
The agent could barely stand after 100 episodes of training.  
![ep1000](https://storage.googleapis.com/gcp_blog/img/ep1000.gif)  
After being trained for 1000 episodes, the agent is able to finish the course (in an inefficient way though).  
![ep2000](https://storage.googleapis.com/gcp_blog/img/ep2000.gif)  
After 2000 episodes of training the agent's gait becomes agile.
