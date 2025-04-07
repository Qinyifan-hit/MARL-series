# The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (2022-MAPPO (with shared parameter))
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
import warnings
import supersuit as ss
from Env_Name import Name, ENV_MAP, Brief_Name

from MAPPO_s_RNN import MAPPO_Agent
from eval import eval_func
from traj import traj_record
from Norm import Normalization

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

dic = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def str2bool(V):
    if isinstance(V, bool):
        return V
    elif V.lower in ('yes', 'true', 't', 'y'):
        return True
    elif V.lower in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mapping(a, up, low):
    a = (up - low) * a + low
    return a


parser = argparse.ArgumentParser()
parser.add_argument("--env_Index", type=int, default=6, help="Index of the environment to use")
parser.add_argument("--max_steps", type=int, default=25, help="Maximum steps per episode")

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(5e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')

parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--clip_rate", type=float, default=0.2, help="GAE parameter")
parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")

parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--load_Idex', type=int, default=int(2e6), help='Index of model for loading')
opt = parser.parse_args()
opt.algo_name = 'MAPPO_s'
opt.apply_padding = False
print(opt)


def main():
    opt.N = 2
    opt.env_name = Name[opt.env_Index]
    opt.BName = Brief_Name[opt.env_Index] + '_' + str(opt.N)

    env_train = ENV_MAP[opt.env_name].parallel_env(N=opt.N, max_cycles=opt.max_steps, continuous_actions=True,
                                                   render_mode='human' if opt.render else None)
    env_eval = ENV_MAP[opt.env_name].parallel_env(N=opt.N, max_cycles=opt.max_steps, continuous_actions=True,
                                                  render_mode='human' if opt.render else None)
    if opt.apply_padding:
        env_train = ss.pad_observations_v0(env_train)
        env_train = ss.pad_action_space_v0(env_train)

        env_eval = ss.pad_observations_v0(env_eval)
        env_eval = ss.pad_action_space_v0(env_eval)

    env_train.reset()
    opt.agents = env_train.agents
    opt.Num_agent = len(opt.agents)
    opt.action_dim = [env_train.action_space(agent).shape[0] for agent in opt.agents]
    opt.obs_dim = [env_train.observation_space(agent).shape[0] for agent in opt.agents]
    opt.state_dim = np.sum(opt.obs_dim)

    opt.action_up = [env_train.action_space(agent).high[0].item() for agent in opt.agents]
    opt.action_low = [env_train.action_space(agent).low[0].item() for agent in opt.agents]

    print('Algorithm:', opt.algo_name, '  Env:', opt.env_name, '  state_dim:', opt.state_dim, ' control range:',
          [opt.action_low, opt.action_up], '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed,
          '  max_e_steps:', opt.max_steps, '\n')

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
    model = MAPPO_Agent(**vars(opt))
    if opt.Loadmodel: model.load(opt.BName, opt.load_Idex)
    Traj = traj_record(opt.batch_size, opt.obs_dim, opt.action_dim, opt.agents, opt.max_steps)

    opt.reward_norm = Normalization(shape=opt.Num_agent)

    if opt.render:
        while True:
            env_show = ENV_MAP[opt.env_name].parallel_env(N=opt.N, max_cycles=opt.max_steps, continuous_actions=True,
                                                          render_mode='human' if opt.render else None)
            score = eval_func(env_show, model, 3, opt.agents)
            print(f'Env:{opt.BName}, seed:{opt.seed}, Episode Agent Reward:{score}, Sum Reward:{sum(score)}')
    else:
        steps = 0
        while steps <= opt.Max_train_steps:
            obs, _ = env_train.reset(seed=env_seed)
            env_seed += 1
            done = False
            counter = 0
            while not done:
                obs_list = [np.array(obs[agent], dtype=np.float32) for agent in opt.agents]
                action_p_list, log_probs_list, action_list = [], [], []
                for agent_id in range(opt.Num_agent):
                    a_p, log_probs = model.action_select(obs_list[agent_id], False)
                    a = mapping(a_p, opt.action_up[agent_id], opt.action_low[agent_id])
                    action_p_list.append(a_p)
                    log_probs_list.append(log_probs)
                    action_list.append(a)
                action = {opt.agents[a_id]: action_list[a_id] for a_id in range(opt.Num_agent)}

                state = torch.cat([torch.FloatTensor(obs[agent]) for agent in opt.agents], dim=-1)
                value_list = model.get_value(state)

                obs_, reward, dw, tr, _ = env_train.step(action)

                obs_list_ = [np.array(obs_[agent], dtype=np.float32) for agent in opt.agents]
                reward_list = [np.array(reward[agent], dtype=np.float32) for agent in opt.agents]
                dones = {agent: dw[agent] or tr[agent] for agent in opt.agents}
                done_list = [np.array(dones[agent], dtype=np.bool_) for agent in opt.agents]
                state_ = torch.cat([torch.FloatTensor(obs_[agent]) for agent in opt.agents], dim=-1)
                value_list_ = model.get_value(state_)

                Traj.update(obs_list, action_p_list, reward_list, obs_list_, state, state_, done_list, log_probs_list, value_list, value_list_, counter)

                done = any(done_list)
                obs = obs_
                steps += 1
                counter += 1

                if Traj.size == int(opt.batch_size):
                    model.train(Traj)
                    Traj.size = 0

                if steps % opt.eval_interval == 0 or steps == 1:
                    score = eval_func(env_eval, model, 3, opt.agents)
                    if opt.write:
                        writer.add_scalar('ep_r', sum(score), global_step=steps)
                    print('Env:', opt.BName, 'seed:', opt.seed, 'steps: {}k'.format(int(steps / 1000)), 'agent_score:',
                          score, 'total_score', sum(score))

                if steps % opt.save_interval == 0:
                    model.save(opt.BName, steps)

    env_show = ENV_MAP[opt.env_name].parallel_env(N=opt.N, max_cycles=opt.max_steps, continuous_actions=True,
                                                  render_mode='human')
    score = eval_func(env_show, model, 10, opt.agents)
    print(f'Env:{opt.BName}, seed:{opt.seed}, Episode Reward:{score}')
    env_show.close()
    env_train.close()
    env_eval.close()


if __name__ == '__main__':
    main()

