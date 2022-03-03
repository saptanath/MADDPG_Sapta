import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from maddpg_env import radioenv
from colorama import Fore, Back, Style
import time 
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from cycler import cycler
import itertools

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def log_FileStart(agents):
        logFile = open("./logs/log.csv", 'w', newline='', encoding='UTF8')
        writer = csv.writer(logFile);    
        header1 = []
        header2 = []
        for index in range(agents):
            header1.append('UE' + str(index))
            header1.append('')
            header2.append('TX power')
            header2.append('Throughput')
        header1.append('All UEs')
        header2.append('Final for perm')
        header1.append('')
        header2.append('episode success')
        header1.append('')
        header2.append('Time to success')
        writer.writerow(header1)
        writer.writerow(header2)
        logFile.close()
        
def log_steps(line):
    logFile = open("./logs/log.csv", 'a', newline='', encoding='UTF8')
    writer = csv.writer(logFile);
    writer.writerow(line)
    logFile.close()
    
def plot_measurements(throughput, reward, episode, i_step):
    plt_through = []
    for index in throughput:
        plt_through.append(throughput[index])
    # Do some nice plotting here
    fig = plt.figure(figsize=(100,40))
    default_cycler = (cycler(color=['r', 'm']) *
                  cycler(linestyle=['-', ':', '-.']))

    plt.rc('lines', linewidth=4)
    plt.rc('axes', prop_cycle=default_cycler)
    
    plt.rc('font', family='serif')

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 50
    # matplotlib.rcParams['text.latex.preamble'] = [
    #     r'\usepackage{amsmath}',
    #     r'\usepackage{amssymb}']    

    plt.xlabel('Episode count')
    
    # Only integers                                
    ax = fig.gca()
    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
    

    start_min = 0
    ax.xaxis.set_ticks(np.arange(start_min, (i_step+2)*10, 10))
    
    #ax_sec = ax.twinx()
    
    ax.set_autoscaley_on(False)
    #ax_sec.set_autoscaley_on(False)
    
    for i, d in enumerate(default_cycler):
        ax.plot(throughput[i], linestyle=d['linestyle'], color=d['color'], label='UE ' + str(i+1))
    
   
    #ax_sec.plot(reward, linestyle='--', color='g', label='Reward')

    
    ax.set_xlim(xmin=start_min, xmax=(i_step+2)*10)
    

    ax.axhline(y=40, xmin=0, color="yellow",  linewidth=1.5, label='throughput target')
    ax.set_ylabel('UE throughput')
    #ax_sec.set_ylabel('Reward')
    
    flat_list = [item for sublist in plt_through for item in sublist]
    
    max_sinr = max(flat_list)
    min_sinr = min(flat_list)
    ax.set_ylim(min_sinr-1, max_sinr + 1)
    #min_rw = min(reward)
    #max_rw = max(reward)
    #ax_sec.set_ylim(min_rw-1, max_rw)
    
    ax.legend(loc="lower right")
    #ax_sec.legend(loc='upper right')
    
    plt.title('Episode {0}'.format(episode))
    plt.grid(True)
    plt.tight_layout()

    
    plt.savefig('figures/measurements_episode_{}.pdf'.format(i_step), format="pdf")
    plt.show(block=True)
    plt.close(fig)

if __name__ == '__main__':
    start_time = time.time()
    end_time = ''
    PRINT_INTERVAL = 50
    N_GAMES = 1000000
    MAX_STEPS = 500
    total_steps = 0
    score_history = []
    evaluate = False
    
    
    env = radioenv()
    n_agents = env.ue_amount
    n_actions = env.resource_blocks
    dims = (n_actions*3) +3
    actor_dims = [dims] * n_agents

    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    

    log_FileStart(n_agents)
     
    
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                            n_actions, n_agents, batch_size=1024)
    
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                               fc1=64, fc2=64,  
                               alpha=0.01, beta=0.01, scenario='maddpg_ul',
                               chkpt_dir='temp/')   


    a = np.identity(n_actions, dtype=np.int32).tolist()
    permutation = list(itertools.permutations(a))
    xCords = [28, 0,-468, 450,-600,-1200] #[28,0,450,-468,-600] # distance = 100, 500, 1000
    yCords = [-96,-250, 176,-600, 800, 0]#[-96,-250,176, 176, 800]
    permutation_cords = list(itertools.permutations(range(n_agents)))
    xcords_ = [0]*n_agents
    ycords_ = [0]*n_agents
    
    pl_through = {ind:[] for ind in range(n_agents)}
    pl_reward = []
    
    for i in range(N_GAMES+1):
        currentperm = np.array(permutation[np.random.randint(low=0, high=len(permutation))])
        currentperm[currentperm==0] = 100
        currentperm = list(currentperm.tolist())
        
        currentperm_cords = permutation_cords[np.random.randint(low=0, high=len(permutation_cords))]
        
        
        for cords in range(len(currentperm_cords)):
            xcords_[cords] = xCords[currentperm_cords[cords]]
            ycords_[cords] = yCords[currentperm_cords[cords]]
        
        explore = True
        load = False 
        
        
    
        obs = env.Reset(maddpg_agents.agents, currentperm, xcords_, ycords_)
        score = 0
        #done = [False]*n_agents
        success = [False]*n_agents
        episode_step = 0
        
        
        while episode_step < 20:
            
            # if load:
            #     maddpg_agents.load_checkpoint()
            episode_step += 1
            
            # actions = maddpg_agents.choose_action(obs)
            # obs_, reward, done, info = env.step(actions)
            actions, explore = maddpg_agents.choose_action(obs, load, n_agents)
            obs_, reward, done, abort, success = env.step(actions, maddpg_agents.agents)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)


            #if episode_step >= MAX_STEPS:

                #success = [True]*n_agents
                
            

            actions_ = []
            for j in range(len(actions)):
                ls = []
                for k in range(len(actions[j])):
                    if round((actions[j][k])) > 0:
                        ls.append(1)
                    else:
                        ls.append(0)
                actions_.append(np.array(ls))

            memory.store_transition(obs, state, actions_, reward, obs_, state_, success)
            
            if episode_step != 0:
                for index, UE in enumerate(maddpg_agents.agents):

                    pl_through[index].append(UE.throughput/1000000)
                pl_reward.append(reward[0])
            
            if (all(done)):
                log_line = []
                if not load and not explore:
                    
                    end_time = time.time() - start_time
                if not explore:
                    
                    maddpg_agents.save_checkpoint()
                    load = True 
                    array = []
                    through = []
                    for index, UE in enumerate(maddpg_agents.agents):
                        array.append(UE.tx_powers)
                        through.append(UE.throughput/1000000)
                        log_line.extend([UE.tx_powers, UE.throughput/1000000])
                    log_line.append('SUCCESS')
                    log_line.append(i)
                    log_line.append(end_time)
                    log_steps(log_line)
                    print(Fore.GREEN + 'SUCCESS.  None of the UEs aborted, best perm = {}. With reward {}. and UE throughput {}'.format((array), (reward[0]), (through)))
                    print(Style.RESET_ALL) 


            if total_steps % 10 == 0:
                maddpg_agents.learn(memory)

            obs = obs_

            score += reward[0]
            total_steps += 1
            
            
            
            # array = []
            # through = []
            # for index, UE in enumerate(maddpg_agents.agents):
            #     array.append(UE.tx_powers)
            #     through.append(UE.throughput/1000000)
            # print('perm = {}. With reward {}. and UE throughput {}, success list {}'.format((array), (reward[0]), (through), (success)))
            # time.sleep(3)
       
            
       
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])
        #plot = threading.Thread(target=plot_measurements, args=(pl_through, pl_reward, i, ))
        #plot.start()
        plot_measurements(pl_through, pl_reward, episode_step, i)
        
        
        if i % PRINT_INTERVAL == 0 and i > 0:
            #load = True 
            #maddpg_agents.save_checkpoint()
            # array = []
            # through = []
            # for index, UE in enumerate(maddpg_agents.agents):
            #     array.append(UE.tx_powers)
            #     through.append(UE.throughput/1000000)

            # print('perm = {}. With reward {}. and UE throughput {}'.format((array), (reward[0]), (through)))

            
            print('episode', i, 'average score {:.10f}'.format(avg_score))
        #del maddpg_agents
