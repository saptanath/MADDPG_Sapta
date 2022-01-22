# Multi-Agent-Deep-Deterministic-Policy-Gradients
A Pytorch implementation of the multi agent deep deterministic policy gradients(MADDPG) algorithm the main code bases was provided for in an open GitHub repo. 

I modified the environemnt and Agent class to work for UL resource block scenario. 

You can find the paper here:
https://arxiv.org/pdf/1706.02275.pdf

You can run on local PC by cloning the repo, and running the main.py. 
Varaibles that need to be changed 
1. NUMBER_OF_UE = 5; found in maddpg_env.py
2. self.resource_blocks = 5, found in maddpg_env.py
3. xCords = [28,-468,-600, 200, 400, 600] # distance = 100, 500, 1000
4. yCords = [-96,176,800, 300, -300, -600] the cords can be found in maddpg_env.py and they need to match the number of UEs. 
5. actor_dims = [12] * n_agents change the value of [x] in main.py to be equal to (number of agents * 2) + 2
