import random
import numpy as np
from importlib import reload
from tqdm import tqdm

import matplotlib.pyplot as plt
import wandb
import torch

from Envs.AYS import AYS_Environment as ays
from Learning import agents as ag
from Learning import utils

def learning_loop(AGENT="A2C",
                  UPDATE_ONLINE=False,
                  BATCH_SIZE=128,
                  MAX_EPISODES=2000,
                  RANDOM_EXPERIENCE=0,
                  LEARNING_RATE=3e-4,
                  BUFFER_SIZE=2**14,
                  REWARD_TYPE='PB',
                  DT=1,
                  SEED=0,
                  MAX_STEPS=600,
                  DISCOUNT=0.999,
                  MAX_FRAMES=1e5,
                  NAME=None,
                  PER_IS=False,
                  PLOT=True,
                  JOB_TYPE=None,
                  EPSILON=0.01,
                  wandbsave=False):

    config = {
        "learning_rate": LEARNING_RATE,
        "max_episodes": MAX_EPISODES,
        "batch_size": BATCH_SIZE,
        "online_updating": UPDATE_ONLINE,
        "random_experience": RANDOM_EXPERIENCE,
        "buffer_size": BUFFER_SIZE,
        "reward_type": REWARD_TYPE,
        "dt": DT,
        "seed": SEED,
        "discount": DISCOUNT,
        "PER_IS": PER_IS,
        "epsilon": EPSILON
    }

    # seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Environment
    reload(ays)
    env = ays.AYS_Environment(discount=DISCOUNT, dt=DT, reward_type=REWARD_TYPE)
    state_dim = len(env.observation_space)
    action_dim = len(env.action_space)

    # Agent
    reload(ag)
    if AGENT=="A2C" or AGENT=="PPO": # Actor critic agents
        agent = eval("ag." + AGENT)(state_dim, action_dim, gamma=DISCOUNT, lr=LEARNING_RATE, epsilon=EPSILON)
    else: # DQN agent
        agent = eval("ag." + AGENT)(state_dim, action_dim, gamma=DISCOUNT, lr=LEARNING_RATE, epsilon=EPSILON)

    # Memory
    if PER_IS:
        memory = utils.PER_IS_ReplayBuffer(capacity=BUFFER_SIZE, alpha=0.6)
    else:
        memory = utils.ReplayBuffer(BUFFER_SIZE)

    # WandB naming
    group = REWARD_TYPE
    job_type = AGENT + "_" + str(LEARNING_RATE) + "_" + str(DISCOUNT) if JOB_TYPE is None else JOB_TYPE
    if PER_IS:
        job_type += "_PER_IS"
    name = "batch" + str(BATCH_SIZE) + "_buffer" + str(BUFFER_SIZE) + "_seed" + str(SEED) if NAME is None else NAME
    if wandbsave:
        wandb.init(group=group,
                   job_type=job_type,
                   config=config,
                   name=name,
                   entity="climate_policy_optim",
                   project="AYS_learning")

    # initialise variables and lists
    rewards = []
    mean_rewards = []
    frame_idx = 0

    # start loop
    for episodes in range(MAX_EPISODES):
        # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
        state = env.reset()
        episode_reward = 0
        I = 1

        for i in range(MAX_STEPS):
            # random experience can help with learning
            if episodes >= RANDOM_EXPERIENCE:
                action = agent.get_action(state)
            else:
                action = np.random.choice(action_dim)

            # step through environment
            next_state, reward, done, = env.step(action)

            # add reward
            episode_reward += reward

            # we can update online or offline
            if UPDATE_ONLINE:
                loss = agent.online_update((state, action, reward, next_state, done), I)
                I *= agent.gamma
                if wandbsave:
                    wandb.log({'loss': loss})
            else:
                memory.push(state, action, reward, next_state, done)
                if memory.__len__() > BATCH_SIZE:
                    if PER_IS:
                        beta = 1 - 0.6 * np.exp(-0.005 * episodes)  # we converge beta to 1
                        sample = memory.sample(BATCH_SIZE, beta)
                        loss, tds = agent.update((sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']), weights=sample['weights'])
                        new_tds = np.abs(tds.cpu().numpy()) + 1e-6  # compatibility
                        memory.update_priorities(sample['indexes'], new_tds)
                    else:
                        sample = memory.sample(BATCH_SIZE)
                        loss, _ = agent.update(sample)
                    if wandbsave:
                        wandb.log({'loss': loss})

            # prepare for next iteration
            state = next_state
            frame_idx += 1

            # if the episode is finished we stop there
            if done:
                break

        # bookkeeping
        rewards.append(episode_reward)
        mean = np.mean(rewards[-50:])
        mean_rewards.append(mean)
        if wandbsave:
            wandb.log({'episode_reward': episode_reward, "moving_average": mean})
        #print("Episode:", episodes, "|| Reward:", round(episode_reward),"|| Final State ", env._which_final_state().name)

        # for notebook
        if PLOT and episodes % 10==0:
            utils.plot(frame_idx, mean_rewards)
            if episodes % 500 == 0:
                utils.plot_test_trajectory(env, agent)

        # if we spend a long time in the simulation
        if frame_idx > MAX_FRAMES:
            break

    # log and show final trajectory
    if wandbsave:
        wandb.run.summary["mean_reward"] = np.mean(rewards)
        wandb.run.summary["top_reward"] = max(rewards)
        wandb.finish()

    if PLOT:
        utils.plot_test_trajectory(env, agent)
    # else:
    #     return np.mean(rewards)
    return agent


def learning_loop_wandb_hparam(config=None, MAX_FRAMES = 1e5, MAX_EPISODES = 2000, MAX_STEPS = 600, agent_str="A2C", per_is=False, seed=0):

    with wandb.init(config=config):

        # variable unpacking
        config = wandb.config

        discount = config.discount
        lr = config.lr
        epsilon = config.epsilon


        if agent_str == "DuellingDQN":
            buffer_size = config.buffer_size
            random_exp = config.random_exp
            batch_size = config.batch_size

            if per_is:
                alpha_buffer = config.alpha_buffer
                beta_buffer = config.beta_buffer

                # Memory
                memory = utils.PER_IS_ReplayBuffer(capacity=buffer_size, alpha=alpha_buffer)
            else:
                memory = utils.ReplayBuffer(buffer_size)

        # seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        reload(ays)
        env = ays.AYS_Environment(discount=discount,)
        state_dim = len(env.observation_space)
        action_dim = len(env.action_space)

        # Agent
        reload(ag)
        if agent_str=="A2C" or agent_str=="PPO": # Actor critic agents
            agent = eval("ag." + agent_str)(state_dim, action_dim, gamma=discount, lr=lr, epsilon=epsilon)
        else: # DQN agent
            agent = eval("ag." + agent_str)(state_dim, action_dim, gamma=discount, lr=lr)

        # initialise variables and lists
        rewards = []
        frame_idx = 0

        # start loop
        for episodes in range(MAX_EPISODES):
            # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
            state = env.reset()
            episode_reward = 0
            I = 1

            for i in range(MAX_STEPS):
                # random experience can help with learning
                if agent_str == "DuellingDQN" and episodes > random_exp:
                    action = agent.get_action(state)
                elif agent_str == "DuellingDQN":
                    action = np.random.choice(action_dim)
                else:
                    action = agent.get_action(state)

                # step through environment
                next_state, reward, done, = env.step(action)

                # add reward
                episode_reward += reward

                if agent_str == "A2C":
                    loss = agent.online_update((state, action, reward, next_state, done), I)
                    I *= discount
                    wandb.log({'loss': loss})
                else:
                    memory.push(state, action, reward, next_state, done)
                    if memory.__len__() > batch_size:
                        if per_is:
                            beta_buffer_t = 1 - beta_buffer * np.exp(-0.005 * episodes)  # we converge beta to 1
                            sample = memory.sample(batch_size, beta_buffer_t)
                            loss, tds = agent.update((sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']), weights=sample['weights'])
                            new_tds = np.abs(tds.cpu().numpy()) + 1e-6  # compatibility
                            memory.update_priorities(sample['indexes'], new_tds)
                        else:
                            sample = memory.sample(batch_size)
                            loss, _ = agent.update(sample)
                        wandb.log({'loss': loss})

                # prepare for next iteration
                state = next_state
                frame_idx += 1

                # if the episode is finished we stop there
                if done:
                    break

            # bookkeeping
            rewards.append(episode_reward)
            wandb.log({"reward": episode_reward})
            wandb.log({"moving_average_reward": np.mean(rewards[-50:])})
            # if we spend a long time in the simulation
            if frame_idx > MAX_FRAMES:
                break

        wandb.log({'mean_reward': np.mean(rewards)})


if __name__ == "__main__":
    learning_loop(AGENT="A2C",
                  SEED=0,
                  UPDATE_ONLINE=True,
                  NAME=None,
                  JOB_TYPE="random",
                  PER_IS=True,
                  BUFFER_SIZE=2 ** 14,
                  BATCH_SIZE=64,
                  BETA=0.01,
                  LEARNING_RATE=1e-4,
                  MAX_EPISODES=2000,
                  MAX_FRAMES=1e5
                  )
    #wandb.agent(entity="1f1ehogx", function=learning_loop_wandb_hparam)
