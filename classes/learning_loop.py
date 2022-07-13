import random
import numpy as np
from importlib import reload
from tqdm import tqdm

import matplotlib.pyplot as plt
import wandb
import torch

from .Envs.AYS import AYS_Environment as ays
from .Learning import agents as ag
from .Learning import utils


def learning_loop(AGENT="PPO",
                  UPDATE_ONLINE=False,
                  BATCH_SIZE=128,
                  MAX_EPISODES=2000,
                  RANDOM_EXPERIENCE=0,
                  LEARNING_RATE=3e-4,
                  BUFFER_SIZE=10000,
                  REWARD_TYPE='PB',
                  DT=1,
                  SCHEDULER=(False, 1000, 0.5),
                  SEED=0,
                  MAX_STEPS=600,
                  DISCOUNT=0.99,
                  MAX_FRAMES=1e5):
    config = {
        "learning_rate": LEARNING_RATE,
        "max_episodes": MAX_EPISODES,
        "batch_size": BATCH_SIZE,
        "online_updating": UPDATE_ONLINE,
        "random_experience": RANDOM_EXPERIENCE,
        "buffer_size": BUFFER_SIZE,
        "reward_type": REWARD_TYPE,
        "dt": DT,
        "scheduler": SCHEDULER,
        "seed": SEED,
        "discount": DISCOUNT,
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
    memory = utils.ReplayBuffer(BUFFER_SIZE)
    agent = eval("ag." + AGENT)(state_dim, action_dim, gamma=DISCOUNT, alpha=LEARNING_RATE)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=SCHEDULER[1], gamma=SCHEDULER[2])

    # WandB naming
    group = REWARD_TYPE
    job_type = AGENT + "_" + str(LEARNING_RATE) + "_" + str(DISCOUNT)
    name = "batch" + str(BATCH_SIZE) + "_buffer" + str(BUFFER_SIZE) + "_seed" + str(SEED)
    wandb.init(group=group,
               job_type=job_type,
               config=config,
               name=name,
               entity="climate_policy_optim",
               project="AYS_learning")

    rewards = []
    mean_rewards = []
    frame_idx = 0

    for episodes in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0

        for i in range(MAX_STEPS):
            if episodes > RANDOM_EXPERIENCE:
                action = agent.get_action(state)
            else:
                action = np.random.choice(action_dim)

            next_state, reward, done, = env.step(action)

            episode_reward += reward

            if UPDATE_ONLINE:
                loss = agent.online_update((state, action, reward, next_state, done))
                wandb.log({'loss': loss})

            else:
                memory.push(state, action, reward, next_state, done)
                if memory.__len__() > BATCH_SIZE:
                    sample = memory.sample(BATCH_SIZE)
                    loss = agent.update(sample)
                    wandb.log({'loss': loss})

            state = next_state
            frame_idx += 1
            if done:
                break

        rewards.append(episode_reward)
        mean_rewards.append(np.mean(rewards[-50:]))

        wandb.log({'episode_reward': episode_reward})
        print("Episode:", episodes, "reward:", round(episode_reward),"|| Final State ",  env._which_final_state().name)

        if not UPDATE_ONLINE and SCHEDULER[0]:
            scheduler.step()

        if frame_idx > MAX_FRAMES:
            break

    wandb.run.summary["mean_reward"] = np.mean(rewards)
    wandb.run.summary["top_reward"] = max(rewards)
    wandb.finish()
    utils.plot_test_trajectory(env, agent)


if __name__ == "__main__":
    learning_loop(AGENT="DuellingDQN",
                  SEED=1)
