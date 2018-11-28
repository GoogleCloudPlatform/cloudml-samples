# Reinforcement Learning using Cloud ML Engine

We present examples of running reinforcement learning (RL) algorithms on GCP
using Cloud ML Engine, specifically, we provide implementations of [DDPG](https://arxiv.org/abs/1509.02971) and [TD3](https://arxiv.org/abs/1802.09477), train and tune their hyper-parameters in the [BipedalWalker](https://gym.openai.com/envs/BipedalWalker-v2) environment.

## Learning curves from hyper-parameter tuning
![Learning curves](img/learning_curve.png?raw=true)  
![Optimal curve](img/optimal_curve.png?raw=true)  

## Learnt TD3 agent
![ep100](img/ep100.gif?raw=true)  
The agent could barely stand after 100 episodes of training.  
![ep1000](img/ep1000.gif?raw=true)
After being trained for 1000 episodes, the agent is able to finish the course.  
(in an inefficient way though)  
![ep2000](img/ep2000.gif?raw=true)  
After 2000 episodes of training the agent's gait becomes agile.
