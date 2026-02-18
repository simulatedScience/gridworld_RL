# Plan
* create a customizable gridworld RL environment
  * save starting positions
  * modular reward functions
  * easily add new special objects
  * visualization
  * parallelization support for sb3 or sbx

* create framework to document and compare agents and reward functions
  * extensive logging of agent behaviour and performance
  * performance measures independent of agents 
    * training steps to reach >`threshold` % success rate
    * number of steps in episode to reach goal
    * rate of exploration
  * easily modify and document agent architecture and hyperparameters
  * visualization of agent behaviour and its evolution during training
    * arrows showing the policy, animated to show policy evolution during training
    * heatmaps of state visitation frequency, reward distribution, etc.

I am particularly interested in exploring ways to make exploration more efficient.
