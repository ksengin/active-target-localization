import torch
import gym
import numpy as np
from target_localization.models.td3 import TD3
import argparse
import os
import matplotlib.pyplot as plt

def eval_rl_agent_strategy(args, env, policy, ep=0):
    """ Function to evaluate the rl agent in a single episode
    """
    ep_reward = 0
    num_targets = args.num_targets if args.test_num_targets is None else args.test_num_targets
    errors = np.zeros((args.num_iters, num_targets))
    uncertainties = np.zeros(args.num_iters)
    state = env.reset()
    target_pos = env._get_true_target_position().numpy()

    for t in range(args.num_iters):
        action = policy.select_action(state)
        state, reward, done, _ = env.step(action)
        ep_reward += reward

        if not args.no_render:
            env.render()

        predictions = env.predictions.numpy()
        errors[t] = np.linalg.norm(target_pos - predictions, ord=2, axis=1)
        uncertainties[t] = env.belief_map.mean()
    print(f'Reward: {ep_reward}')
    return errors, uncertainties

def get_policy(args, env, directory, filename):
    """ Function to load the policy network (and convnet) weights
    """
    observation_space = 1 if env.observation_space.shape == () else env.observation_space.shape[0]
    state_dim = observation_space
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(args.lr, state_dim, action_dim, max_action)
    
    policy.load_actor(directory, filename)
    print(f'Actor used: {directory}/{filename}')
    if args.image_representation:
        env.convnet.load_state_dict(torch.load(args.archive_convnet))
        print(f'Convnet used: {args.archive_convnet}')
    
    return policy, env


def main(args: argparse.Namespace):
    env_name = "TrackingWaypoints-v0"
    random_seed = 0
    n_episodes = args.num_episodes
    lr = args.lr
    render = not args.no_render
    num_targets = args.num_targets
    reward_type = args.reward_type
    fixed_config = False
    image_representation = args.image_representation
    meas_model = args.meas_model
    sess = args.sess
    static_target = not args.dynamic_target
    test_num_targets = num_targets if args.test_num_targets is None else args.test_num_targets
    
    filename = "TD3_{}_{}_m{}_{}".format(env_name, random_seed, num_targets, reward_type)
    directory = f"target_localization/archive/{env_name}/{args.sess}"
    
    env = gym.make(env_name)
    random_seed = 1
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    env.env_parametrization(num_targets=test_num_targets, reward_type=reward_type, image_representation=image_representation, meas_model=meas_model, \
        static_target=static_target, augment_state=not args.no_augmented_state)
    policy, env = get_policy(args, env, directory, filename)
    
    for ep in range(n_episodes):
        eval_rl_agent_strategy(args, env, policy, ep=ep)
        print(f'Episode : {ep}')
                
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_targets', type=int, default=2)
    parser.add_argument('--num_iters', type=int, default=50)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--reward_type', type=str, default='heatmap')
    parser.add_argument('--image_representation', action='store_true')
    parser.add_argument('--sess', type=str)
    parser.add_argument('--archive_convnet', type=str)
    parser.add_argument('--meas_model', type=str, default='bearing')
    parser.add_argument('--save_figs', action='store_true')
    parser.add_argument('--dynamic_target', action='store_true')
    parser.add_argument('--no_augmented_state', action='store_true')
    parser.add_argument('--test_num_targets', type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(get_args())
    
