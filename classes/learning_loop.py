import random
import numpy as np
from importlib import reload
from tqdm import tqdm

import wandb
import torch

import Envs.AYS.AYS_Environment as ays
import Learning.agents as ag
import Learning.utils as utils

AGENT = "PPO"
UPDATE_ONLINE = False
BATCH_SIZE = 128
MAX_EPISODES = 2000
RANDOM_EXPERIENCE = 0
LEARNING_RATE = 3e-4
BUFFER_SIZE = 10000
REWARD_TYPE = 'policy_cost'
DT = 1
SCHEDULER = (False, 1000, 0.5)
SEED = 0
MAX_STEPS = 600

# config
default_config = {
    "learning_rate": LEARNING_RATE,
    "max_episodes": MAX_EPISODES,
    "batch_size": BATCH_SIZE,
    "online_updating": UPDATE_ONLINE,
    "random_experience": RANDOM_EXPERIENCE,
    "buffer_size": BUFFER_SIZE,
    "reward_type": REWARD_TYPE,
    "dt": DT,
    "scheduler": SCHEDULER,
    "seed": SEED
}

class LearningLoop:
    def __init__(self, config=None):

        if config is None:
            config = default_config
        # seeds
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Environment
        reload(ays)
        env = ays.AYS_Environment(dt=DT, reward_type=REWARD_TYPE)
        state_dim = len(env.observation_space)
        action_dim = len(env.action_space)

        # Agent
        reload(ag)
        memory = utils.ReplayBuffer(BUFFER_SIZE)
        agent = eval("ag." + AGENT)
        agent = agent.__init__(state_dim, action_dim, alpha=LEARNING_RATE)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=SCHEDULER[1], gamma=SCHEDULER[2])

        group = AGENT + "_" + REWARD_TYPE + "_" + str(LEARNING_RATE)
        job_type = "batch" + str(BATCH_SIZE) + "_buffer" + str(BUFFER_SIZE) + "_seed" + str(SEED)
        wandb.init(group=group,
                   config=config,
                   job_type=job_type,
                   entity="climate_policy_optim",
                   project="AYS_learning")

        rewards = []
        mean_rewards = []
        frame_idx = 0

        for episodes in tqdm(range(MAX_EPISODES)):
            state = env.reset()
            episode_reward = 0
            done = False

            for i in range(MAX_STEPS):
                if episodes > RANDOM_EXPERIENCE:
                    action = agent.get_action(state)
                else:
                    action = np.random.choice(action_dim)

                next_state, reward, done, = env.step(action)

                if done:
                    reward += env.calculate_expected_final_reward(agent.gamma, MAX_STEPS, current_step=i)
                    # if env._good_final_state():
                    #     reward += 100
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

            # if not UPDATE_ONLINE:
            if SCHEDULER[0]:
                scheduler.step()

            wandb.log({'episode_reward': episode_reward})
            if episodes % 200 == 0:
                utils.plot_test_trajectory(env, agent)

            else:
                if episodes % 10 == 0:
                    utils.plot(frame_idx, mean_rewards)
                    if episodes % 30 == 0:
                        utils.plot_test_trajectory(env, agent)

        wandb.run.summary["mean_reward"] = np.mean(rewards)
        wandb.run.summary["top_reward"] = max(rewards)
        wandb.finish()
