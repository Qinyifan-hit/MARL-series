# Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (2017-MADDPG)
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
import warnings
import supersuit as ss
from Env_Name import Name, ENV_MAP, Brief_Name

from MADDPG import DDPG_Agent
from Buffer import Replay_Buffer
from eval import eval_func

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def str2bool(V):
    if isinstance(V, bool):
        return V
    elif V.lower in ('yes', 'true', 't', 'y'):
        return True
    elif V.lower in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--env_Index", type=int, default=6, help="Index of the environment to use")
parser.add_argument("--total_steps", type=int, default=int(2e6), help="Total time steps")
parser.add_argument("--buffer_size", type=int, default=int(2e6), help="Replay buffer size")
parser.add_argument("--warmup_steps", type=int, default=20000, help="Warmup steps")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--max_steps", type=int, default=25, help="Maximum steps per episode")
parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter")
parser.add_argument("--a_lr", type=float, default=1e-3, help="Actor learning rate")
parser.add_argument("--c_lr", type=float, default=2e-3, help="Critic learning rate")
parser.add_argument("--hidden_sizes", type=int, default=int(256), help="Hidden layer sizes (comma-separated)")
parser.add_argument("--update_every", type=int, default=15, help="Update networks every n steps")
parser.add_argument("--noise_scale", type=float, default=0.3, help="Initial noise scale")
parser.add_argument("--min_noise", type=float, default=0.05, help="Minimum noise scale")
parser.add_argument("--noise_decay_steps", type=int, default=int(3e5), help="Number of step to decay noise to min_noise default: 300k")
parser.add_argument("--use_noise_decay", action="store_true", help="Use noise decay")
parser.add_argument("--create_gif", action="store_true", help="Create GIF of episodes")
parser.add_argument("--eval_interval", type=int, default=5000, help="Evaluate every n steps")
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument("--seed", type=int, default=0, help="random seed")

parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=True, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--load_Idex', type=int, default=int(2e6), help='Index of model for loading')
opt = parser.parse_args()
opt.algo_name = 'MADDPG'
opt.apply_padding = False
print(opt)

def main():
    # environment making
    opt.env_name = Name[opt.env_Index]
    opt.BName = Brief_Name[opt.env_Index]

    env_train = ENV_MAP[opt.env_name].parallel_env(N=3, max_cycles=opt.max_steps, continuous_actions=True, render_mode='human' if opt.render else None)
    env_eval = ENV_MAP[opt.env_name].parallel_env(N=3, max_cycles=opt.max_steps, continuous_actions=True, render_mode='human' if opt.render else None)
    if opt.apply_padding:
        env_train = ss.pad_observations_v0(env_train)
        env_train = ss.pad_action_space_v0(env_train)

        env_eval = ss.pad_observations_v0(env_eval)
        env_eval = ss.pad_action_space_v0(env_eval)

    env_train.reset()
    opt.agents = env_train.agents
    opt.Num_agent = len(opt.agents)
    opt.action_dim = [env_train.action_space(agent).shape[0] for agent in opt.agents]
    opt.state_dim = [env_train.observation_space(agent).shape[0] for agent in opt.agents]

    opt.action_up = [env_train.action_space(agent).high[0].item() for agent in opt.agents]
    opt.action_low = [env_train.action_space(agent).low[0].item() for agent in opt.agents]

    print('Algorithm:', opt.algo_name, '  Env:', opt.env_name, '  state_dim:', opt.state_dim, ' control range:', [opt.action_low, opt.action_up],
         '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_steps, '\n')

    env_seed = opt.seed
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(opt.algo_name, opt.BName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    models = []
    for Agent_id in range(opt.Num_agent):
        opt.id = Agent_id
        model = DDPG_Agent(**vars(opt))
        models.append(model)
    if opt.Loadmodel:
        for Agent_id in range(opt.Num_agent):
            models[Agent_id].load(opt.BName, opt.load_Idex, int(Agent_id))
    Reply = Replay_Buffer(opt.action_dim, opt.state_dim, opt.buffer_size, opt.agents, opt.Num_agent)

    opt.noise_decay = (opt.noise_scale - opt.min_noise) / opt.noise_decay_steps

    if opt.render:
        while True:
            env_show = ENV_MAP[opt.env_name].parallel_env(N=3, max_cycles=opt.max_steps, continuous_actions=True, render_mode='human' if opt.render else None)
            score = eval_func(env_show, models, 3, opt.agents)
            print(f'Env:{opt.BName}, seed:{opt.seed}, Episode Agent Reward:{score}, Sum Reward:{sum(score)}')
    else:
        steps = 0
        while steps <= opt.total_steps:
            state, _ = env_train.reset(seed=env_seed)  # dictionary for environment
            done = False
            env_seed += 1
            while not done:
                state_list = [np.array(state[agent], dtype=np.float32) for agent in opt.agents]
                if steps < opt.warmup_steps:
                    action_list = [env_train.action_space(agent).sample() for agent in opt.agents]
                else:
                    action_list = [models[agent_id].action_select(state_list[agent_id], False) for agent_id in range(opt.Num_agent)]
                action = {opt.agents[agent_id]: action_list[agent_id] for agent_id in range(opt.Num_agent)}

                state_, reward, dw, tr, _ = env_train.step(action)

                state_list_ = [np.array(state_[agent], dtype=np.float32) for agent in opt.agents]
                reward_list = [np.array(reward[agent], dtype=np.float32) for agent in opt.agents]
                dw_list = [np.array(dw[agent], dtype=np.bool_) for agent in opt.agents]
                dones = {agent: (dw[agent] or tr[agent]) for agent in opt.agents}
                done_list = [np.array(dones[agent], dtype=np.bool_) for agent in opt.agents]

                Reply.add(state_list, action_list, reward_list, state_list_, done_list, dw_list)

                done = any(done_list)
                state = state_
                steps += 1

                opt.noise_scale = max(opt.min_noise, opt.noise_scale - opt.noise_decay)
                for agent_id in range(opt.Num_agent):
                    models[agent_id].noise = opt.noise_scale


                if steps > opt.warmup_steps and steps % opt.update_every == 0:
                    for agent_id in range(opt.Num_agent):
                        s_b, a_b, r_b, s_b_, done_b = Reply.B_sample(opt.batch_size)
                        a_b_ = [models[a_id].A_target(s_b_[a_id]) for a_id in range(opt.Num_agent)]
                        models[agent_id].train(s_b, a_b, r_b, s_b_, done_b, a_b_, agent_id)

                if steps % opt.eval_interval == 0 or steps == 1:
                    score = eval_func(env_eval, models, 3, opt.agents)
                    if opt.write:
                        writer.add_scalar('ep_r', sum(score), global_step=steps)
                    print('Env:', opt.BName, 'seed:', opt.seed, 'steps: {}k'.format(int(steps/1000)), 'agent_score:', score, 'total_score', sum(score))

                if steps % opt.save_interval == 0:
                    for agent_id in range(opt.Num_agent):
                        models[agent_id].save(opt.BName, steps, agent_id)

    env_show = ENV_MAP[opt.env_name].parallel_env(N=3, max_cycles=opt.max_steps, continuous_actions=True, render_mode='human')
    score = eval_func(env_show, models, 10, opt.agents)
    print(f'Env:{opt.BName}, seed:{opt.seed}, Episode Reward:{score}')
    env_show.close()
    env_train.close()
    env_eval.close()


if __name__ == '__main__':
    main()



