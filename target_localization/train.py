import torch
import gym
import numpy as np
import argparse
import os
import visdom
import json
from target_localization.models.td3 import TD3
from target_localization.util.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

def main(args: argparse.Namespace):
    num_targets = args.num_targets
    reward_type = args.reward_type
    image_representation = args.image_representation

    env_name = "TrackingWaypoints-v0"
    log_freq = args.log_freq
    random_seed = args.seed
    gamma = args.gamma
    batch_size = args.batch_size
    lr = args.lr
    num_episodes = args.num_episodes
    num_iters = args.num_iters
    directory = "target_localization/archive/{}/{}".format(env_name, args.sess)
    filename = "TD3_{}_{}_m{}_{}".format(env_name, random_seed, num_targets, reward_type)

    vis = visdom.Visdom()
    reward_list = []
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, 'session_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if random_seed:
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    env = gym.make(env_name)

    # Initializing the environment parameters
    env.env_parametrization(num_targets=num_targets, reward_type=reward_type, image_representation=image_representation, \
        vis=(vis, args.sess), meas_model=args.meas_model, augment_state=not args.no_augmented_state, im_loss=args.im_loss)
    
    # Dimensions and max action magnitude
    observation_space = 1 if env.observation_space.shape == () else env.observation_space.shape[0]
    state_dim = observation_space
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize the actor critic networks and replay buffer
    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)

    # For fine-tuning pass the previous training folder in the archive_agent argument
    if args.archive_agent is not None:
        archive_dir = "target_localization/archive/{}/{}".format(env_name, args.archive_agent)
        policy.load(archive_dir, f'{filename}')
        if image_representation:
            convnet_path = f'{archive_dir}/{filename}_convnet.pth'
            env.convnet.load_state_dict(torch.load(convnet_path))
    
    mean_reward, ep_reward = 0, 0
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        for t in range(num_iters):
            # use the actor network to select an action and add noise to facilitate exploration
            action = policy.select_action(state) + torch.normal(0, args.exploration_noise, size=env.action_space.shape)
            action = action.clamp(env.action_space.low.item(), env.action_space.high.item())
            
            # take an action in the environment and add the tuple to the replay buffer
            next_state, reward, done, reward_info = env.step(action)
            replay_buffer.add((state, action, reward, next_state, np.float(done)))
            state = next_state
            if args.render:
                env.render()

            mean_reward += reward
            ep_reward += reward
            
            if done:
                break

        # Update the policy by sampling from the replay buffer
        policy.update(replay_buffer, t, batch_size, args.gamma, args.tau, args.policy_noise, args.policy_delay)
        print(f'reward: {ep_reward}')
        ep_reward = 0

        # Save actor critic weights
        if episode > 500 and episode % 10 == 0:
            policy.save(directory, filename)
            if args.image_representation:
                torch.save(env.convnet.state_dict(), f'{directory}/{filename}_convnet.pth')
        
        # Print/display results every log_freq episodes
        if episode > 0 and episode % log_freq == 0:
            reward_list.append(mean_reward.item() / log_freq)
            vis.line(X=np.arange(len(reward_list)), Y=reward_list, win='reward', env=args.sess)
            mean_reward = mean_reward // log_freq
            print(f"Episode: {episode}\tAverage Reward: {mean_reward}")
            mean_reward = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_targets', type=int, default=2)
    parser.add_argument('--num_episodes', type=int, default=4000)
    parser.add_argument('--num_iters', type=int, default=60)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reward_type', type=str, default='heatmap')
    parser.add_argument('--image_representation', action='store_true')
    parser.add_argument('--sess', type=str, default='atl')
    parser.add_argument('--meas_model', type=str, default='bearing')
    parser.add_argument('--no_augmented_state', action='store_true')
    parser.add_argument('--archive_agent', type=str)
    parser.add_argument('--im_loss', action='store_true')
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)

    ## rl algorithm hyperparameters
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount factor')
    parser.add_argument('--tau', type=float, default=0.99, help='Polyak update weight')
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--policy_delay', type=int, default=2, help='Num of iterations policy is frozen')
    parser.add_argument('--exploration_noise', type=float, default=0.1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(get_args())