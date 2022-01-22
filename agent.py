import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np


exploration_rate = 1.0
decay_rate = 0.9995

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.01, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name='_'+self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name='_'+self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name='_'+self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name='_'+self.agent_name+'_target_critic')
        self.x_ue, self.y_ue = 0, 0 # X and Y cordinate of base station 1 = 0,0
        self.tx_powers = []
        self.tx_power_interferer = []  # transmitionPower, pathLoss
        self.sinr = []
        self.throughput = 1
        self.rx = []
        self.exploration_rate = exploration_rate/decay_rate
        self.decay_rate = decay_rate
        self.exploration_min = 0.1
        
        self.lastAction = 0
        self.lastActionRandom = 0
        self.lastActionResult = 0
        self.state_log = []

        self.update_network_parameters(tau=1)
        
 
    def choose_action(self, observation, load, n_agents):
        if not load:
            self.exploration_rate *= self.decay_rate
            if self.exploration_rate < self.exploration_min:
                self.exploration_rate = self.exploration_min
                
            if np.random.uniform(0, 1) > self.exploration_rate:
                action_rn = (np.random.randint(2, size=n_agents))
                return action_rn, True
            else:
                state = T.tensor([observation], dtype=T.float).to(self.actor.device)
                actions = self.actor.forward(state)
                noise = T.rand(self.n_actions).to(self.actor.device)
                action = actions + noise
                return action.detach().cpu().numpy()[0], False
        else:
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            actions = self.actor.forward(state)
            noise = T.rand(self.n_actions).to(self.actor.device)
            action = actions + noise
            return action.detach().cpu().numpy()[0], False

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def reset_states(self, rb, x, y):
       #self.x_ue = self.np_random.uniform(low=-env.cell_radius, high=env.cell_radius)
       #self.y_ue = self.np_random.uniform(low=-env.cell_radius, high=env.cell_radius)
       
       self.x_ue = x
       self.y_ue = y
       
       #self.tx_power = env.pt_serving[self.np_random.randint(low=0, high=TX_POWERLEVELS_COUNT)]
       self.tx_power_interferer = []
       self.sinr = []
       self.rx = []
       self.tx_powers = [0] * rb
       
       for index in np.arange(rb):
         self.rx.append(1)
         self.tx_power_interferer.append(1)
         self.sinr.append(1)
    
    def save_states(self, oldStates):
        #print(oldStates)
        states = []
        states.append(self.x_ue)
        states.append(self.y_ue)
        for index in np.arange(len(self.tx_powers)):
            states.append(oldStates[2+index])
        states.append(self.lastAction)
        states.append(self.lastActionRandom)
        states.append(self.lastActionResult)
        self.state_log.append(states)
    
    def saveAction(self,action):
        self.lastAction = action
        # self.lastActionRandom = actionRandom
        # if(abort):
        #     self.lastActionResult = "Abort"
        # elif(done):
        #     self.lastActionResult = "Success"
        # else:
        #     self.lastActionResult = "None"
        
    def get_states(self):
        states = []
        states.append(self.x_ue)
        states.append(self.y_ue)
        states.extend(self.tx_powers)
        states.extend(self.tx_power_interferer)
        states.append(self.throughput)
        return states