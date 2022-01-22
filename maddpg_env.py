#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:40:32 2021

@author: saptaparnanath
"""

import numpy as np

import math
import random 



SEED = 1;
NUMBER_OF_UE = 6;
MAX_EPISODES_OPT = 1
FILENAME = "0"

class radioenv(object):
    
    def __init__(self):


        self.x_bs_1, self.y_bs_1 = 0, 0 # X and Y cordinate of base station 1 = 0,0
        self.ue_amount = NUMBER_OF_UE;  # The amount of Upload Equipment
        self.ue = []; # list of UE classes
        self.ue_sinr = []; # list of SINR for each UE
        
        self.radio_frame = 10 # equal to max_timesteps_per_episode
        self.reward_min = -5
        self.reward_max = 15
        
    
        self.episode_loss_list = []
        self.episode_q_list = []
        self.episode_successful_list = [] # a list to save the good episodes
        
        self.observation_list = [None] * NUMBER_OF_UE
        self.next_observation_list = [None] * NUMBER_OF_UE
        self.reward_list = [0] * NUMBER_OF_UE
        self.done_list = [False] * NUMBER_OF_UE
        self.abort_list = [False] * NUMBER_OF_UE
        
        self.success_list = [False] * NUMBER_OF_UE
        self.actions_list = []
        self.currentAction_list = [-1] * NUMBER_OF_UE
        self.currentActionRandom_list = [-1] * NUMBER_OF_UE
        self.total_reward_list = [0] * NUMBER_OF_UE
        self.Q_values_list = []   
        self.losses_list = []
        
        self.sinrGraphData = []
        self.TxPowerGraphData = []
        self.rxGraphData = []
        self.sinr_subChannels = []
    
        self.resource_blocks = NUMBER_OF_UE
        self.resource_block_power_levels = [10] * self.resource_blocks #in dbm
        self.cell_radius = 1000 # in meters.
        self.bandwidth = (self.ue_amount*5)*pow(10, 6) # 20Mhz
        self.resource_blocks_bandwidth = self.bandwidth/self.resource_blocks #5Mhz each
        self.f_c = 3.5e9 # Hz = 3.5GHz 
     
        
        self.logs = []
       

        for index in np.arange(self.ue_amount):
            self.episode_loss_list.append([])
            self.episode_q_list.append([])
            self.Q_values_list.append([])   
            self.losses_list.append([])
            self.actions_list.append([])
            self.rxGraphData.append([])
    
    
    def Reset(self, agents):
        xCords = [28, 0,-468, 450,-600,-1200] #[28,0,450,-468,-600] # distance = 100, 500, 1000
        yCords = [-96,-250, 176,-600, 800, 0]#[-96,-250,176, 176, 800]
        #xCords = [28,0,450,-468,-600] # distance = 100, 500, 1000
        #yCords = [-96,-250,176, 176, 800]
        #xCords = [28,   0,450,-600]
        #yCords = [-96,-250,176, 800]
        for index, UE in enumerate(agents):
            # for each UE begin a new episode   
            self.reset_ue_episode(index)    
            UE.reset_states(self.resource_blocks, xCords[index], yCords[index])

            action = np.random.randint(2, size=self.ue_amount)
            for index_b in np.arange(self.resource_blocks):
                bit = action[index_b]
                if(bit==1):
                    UE.tx_powers[index_b] = self.resource_block_power_levels[index_b]
                else:
                    UE.tx_powers[index_b] = -60 #dbm
            self.observation_list[index] = UE.get_states()

        return self.observation_list
            
    def reset_ue_episode(self, index):
        self.done_list[index] = False # this UE is not done
        self.abort_list[index] = False # this UE is not done
        self.success_list[index] = False;
        self.total_reward_list[index] = 0 # set the total reward to 0
        self.actions_list[index] = [] #set the actions taken to an empty list
        self.episode_loss_list[index] = []
        self.episode_q_list[index] = []
    
    def calculate_recived_transmision_power(self, agents):
        Rx = [0] * self.resource_blocks
        for RB_index in np.arange(self.resource_blocks):
            for index, UE in enumerate(agents): 
                UE.rx[RB_index] = (self._compute_receive(UE,UE.tx_powers[RB_index]))
                Rx[RB_index] += UE.rx[RB_index]
        return Rx
   
    def calculate_ue_interference(self, Rx, agents):
        for RB_index in np.arange(self.resource_blocks):
            for index, UE in enumerate(agents):
                UE.tx_power_interferer[RB_index] = Rx[RB_index] - UE.rx[RB_index]
            
    def calculate_ue_sinr(self, agents):
        noise = self._compute_noise()
        self.sinr_subChannels = []
        B = self.resource_blocks_bandwidth
        for index, UE in enumerate(agents):
            UE.throughput = 0
        for RB_index in np.arange(self.resource_blocks):
            for index, UE in enumerate(agents):
                UE.sinr[RB_index] = UE.rx[RB_index] / (UE.tx_power_interferer[RB_index] + noise)
                #UE.throughput =  B * math.log2( )
                self.sinr_subChannels.append(B * math.log(UE.sinr[RB_index] + 1, 2))
                UE.throughput += B * math.log(UE.sinr[RB_index] + 1, 2)

    def convert_UE_to_dbm(self, agents):
        for RB_index in np.arange(self.resource_blocks):
            for UE_index, UE in enumerate(agents):
                UE.sinr[RB_index] = self._watt_to_dbm(UE.sinr[RB_index])
                UE.tx_power_interferer[RB_index] = self._watt_to_dbm(UE.tx_power_interferer[RB_index])
                UE.rx[RB_index] = self._watt_to_dbm(UE.rx[RB_index])
    
    def calculate_reward(self, agents):
        Rx = self.calculate_recived_transmision_power(agents)
        self.calculate_ue_interference(Rx, agents)
        self.calculate_ue_sinr(agents)
        self.convert_UE_to_dbm(agents) ## for debuging so values are readable
        return self._watt_to_dbm(np.std(self.sinr_subChannels))
    
    def _compute_noise(self): ## may need to add distubution for the noise 
        T = 290 # Kelvins
        B = self.resource_blocks_bandwidth # Hz ## I think this should change to self.resource_blocks_bandwidth
        k_Boltzmann = 1.38e-23
        self.noise_power = k_Boltzmann*T*B # this is in Watts
        return self.noise_power  # return in watts
            

    def _watt_to_dbm(self,watts):
        return 10 * math.log10(1000 * watts)
    
    def _dbm_to_watt(self,dmb):
        return math.pow(10,dmb/10) / 1000

        
    def _compute_receive(self, ue_obj, tx_power):
        
        
        transmitter_Power = tx_power # dBm
        tx_antenna_gain = 0 # dBi as per email
        free_Space_Path_Loss = self._free_space_path_loss(ue_obj.x_ue, ue_obj.y_ue) #dB
        rx_Antenna_Gain = 18 #dBi as per email 
        
        signal = transmitter_Power + tx_antenna_gain - free_Space_Path_Loss + rx_Antenna_Gain
        
        return self._dbm_to_watt(signal) # return signal in watts, cell class will convert if needed
            
    def _return_values(self):
        interference_plus_noise_power = self.mean_int_power + self.noise_power
        self.received_sinr = 10*np.log10(self.received_power / interference_plus_noise_power)

        return [self.received_power, self.mean_int_power, self.received_sinr]
            
    def _free_space_path_loss(self,x,y):
        x_bs, y_bs = 0,0
        f = self.f_c
        d = math.sqrt((x - x_bs)**2 + (y - y_bs)**2) # in meters
        # c = 3e8
        # 147.55 = 20 log10(4*Math.pi() / c)
        return 20 * math.log10(d) + 20 * math.log10(f) - 147.55
    
        
    
    def step(self, actions, agents):
        v = 2 # km/h.

        v *= 5./18 # in m/sec
        theta_1 = np.random.uniform(-math.pi, math.pi, size=1)
        
        dx_1 = v * math.cos(theta_1)
        dy_1 = v * math.sin(theta_1)
        for index, UE in enumerate(agents):
            # self.currentAction_list[index] = UE.act(self.observation_list[index])
            for index_b in np.arange(self.resource_blocks):
                bit = int(actions[index][index_b])

                if bit > 0:
                    UE.tx_powers[index_b] = self.resource_block_power_levels[index_b]
                else:
                    UE.tx_powers[index_b] = -60 #dbm
                # UE.tx_powers[index_b] = self.resource_block_power_levels[index_b]*bit
            UE.x_ue += dx_1
            UE.y_ue += dy_1

        reward = self.calculate_reward(agents)
        for index, UE in enumerate(agents):
            if (UE.throughput/1000000) < 40:

                reward -= 20
        # if not change:
        #     reward = reward/100
        #reward = reward/100
        # abort = False 
        
        for index, UE in enumerate(agents):
            self.reward_list[index] = reward
            self.done_list[index],self.abort_list[index] =  self._done_abort(UE)
            self.total_reward_list[index] += reward   
            self.success_list[index] = (self.done_list[index] and (self.abort_list[index] == False))

            self.next_observation_list[index] = UE.get_states()
            # if(self.abort_list[index] == True):
            #     abort = True 
        return self.next_observation_list, self.reward_list, self.done_list, self.abort_list, self.success_list
            
    
    def _done_abort(self, ue_object):
        
        #allDone = True
        #anyAbort = False
        done = True
        abort = False
        # for index in np.arange(self.resource_blocks):
        
        #     done = (ue_object.tx_powers[index] <= self.max_tx_power) and (ue_object.tx_power_interferer[index] <= self.max_tx_power_interference) and \
        #             (ue_object.sinr[index] >= self.min_sinr) and (ue_object.sinr[index] >= self.sinr_target) #and (received_ue2_sinr >= self.sinr_target)
       
        #     abort = (ue_object.tx_powers[index] > self.max_tx_power) or (ue_object.tx_power_interferer[index] > self.max_tx_power_interference) or (ue_object.sinr[index] < self.min_sinr) \
        #         or (ue_object.sinr[index] > 100)  #or (received_sinr < 10) or (received_ue2_sinr < 10)  
        #     if abort == True:
        #         anyAbort = True
        #         allDone = False
        #     if done == False:
        #         allDone = False
        if (ue_object.throughput/1000000 < 0.2) or (sum(ue_object.tx_powers) == -60*self.resource_blocks):
            abort = True 
            done = False 
        elif (ue_object.throughput/1000000 > 40):

            done = True 
            abort = False
        else:
            done = False
            abort = True 
        return done, abort
        