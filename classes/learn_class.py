from Envs.AYS.AYS_Environment import AYS_Environment
import Learning.agents as ag
import Learning.utils as utils
import random
import numpy as np
import torch
import wandb


class learning:
    def __init__(self, wandb_save=False, verbose=False, reward_type="PB", max_episodes=2000, max_steps=600, max_frames=1e5, max_epochs=200):

        # environment
        self.env = AYS_Environment(reward_type=reward_type)
        self.state_dim = len(self.env.observation_space)
        self.action_dim = len(self.env.action_space)

        # seeds
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # parameters to keep
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.max_frames = max_frames
        self.max_epochs = max_epochs

        # saving in wandb or logging
        self.wandb_save = wandb_save
        self.verbose = verbose

    def learning_loop_online(self, agent_str, notebook=False, plotting=False):

        wandb.init() if self.wandb_save else None
        env = self.env

        if agent_str == "DuelDQN" or agent_str == "DuelDDQN":
            print("wrong agents, use Actor Critics: A2C or PPO")
            return

        agent = self.agent

        rewards = []
        mean_rewards = []
        frame_idx = 0

        for episodes in range(self.max_episodes):

            # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
            state = env.reset()
            episode_reward = 0
            I = 1

            for i in range(self.max_steps):

                # get state
                action = agent.get_action(state)

                # step through environment
                next_state, reward, done, = env.step(action)

                # add reward
                episode_reward += reward

                # update the agent with last experience
                loss = agent.online_update((state, action, reward, next_state, done), I)
                I *= agent.gamma

                wandb.log({'loss': loss}) if self.wandb_save else None

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

            # we log or print depending on settings
            wandb.log({'episode_reward': episode_reward, "moving_average": mean}) if self.wandb_save else None
            print("Episode:", episodes, "|| Reward:", round(episode_reward),"|| Final State ", env._which_final_state().name) if self.verbose else None

            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(frame_idx, mean_rewards)
                if episodes % 500 == 0:
                    utils.plot_test_trajectory(env, agent)

            # if we spend a long time in the simulation
            if frame_idx > self.max_frames:
                break

        # log and show final trajectory
        if self.wandb_save:
            wandb.run.summary["mean_reward"] = np.mean(rewards)
            wandb.run.summary["top_reward"] = max(rewards)
            wandb.finish()

        if plotting:
            utils.plot(frame_idx, mean_rewards)
            utils.plot_test_trajectory(env, agent)
        return agent

    def learning_loop_offline(self, agent_str, buffer_size, batch_size, per_is, notebook=False, plotting=False, alpha=0.6, beta=0.4):

        wandb.init() if self.wandb_save else None
        env = self.env

        # initiate memory
        memory = utils.PER_IS_ReplayBuffer(buffer_size, alpha=alpha) if per_is else utils.ReplayBuffer(buffer_size)

        if agent_str != "DuellingDQN" or agent_str != "DuellingDDQN":
            print("wrong agents, use DQN agents: DuelDQN or DuelDDQN")
            return

        agent = self.agent

        rewards = []
        mean_rewards = []
        frame_idx = 0

        for episodes in range(self.max_episodes):

            # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
            state = env.reset()
            episode_reward = 0

            for i in range(self.max_steps):

                # get state
                action = agent.get_action(state)

                # step through environment
                next_state, reward, done, = env.step(action)

                # add reward
                episode_reward += reward

                memory.push(state, action, reward, next_state, done)

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

            for epoch in range(self.max_epochs):
                if memory.__len__() > batch_size:
                    if per_is:
                        beta = 1 - (1-beta) * np.exp(-0.005 * episodes)  # we converge beta to 1
                        sample = memory.sample(batch_size, beta)
                        loss, tds = agent.update(
                            (sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']),
                            weights=sample['weights']
                        )
                        new_tds = np.abs(tds.cpu().numpy()) + 1e-6
                        memory.update_priorities(sample['indexes'], new_tds)
                    else:
                        sample = memory.sample(batch_size)
                        loss, _ = agent.update(sample)

                    wandb.log({'loss': loss}) if self.wandb_save else None
            print("hello")
            # we log or print depending on settings
            wandb.log({'episode_reward': episode_reward, "moving_average": mean}) if self.wandb_save else None
            print("Episode:", episodes, "|| Reward:", round(episode_reward),"|| Final State ", env._which_final_state().name) if self.verbose else None

            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(frame_idx, mean_rewards)
                if episodes % 500 == 0:
                    utils.plot_test_trajectory(env, agent)

            # if we spend too long in the simulation
            if frame_idx > self.max_frames:
                break

        # log and show final trajectory
        if self.wandb_save:
            wandb.run.summary["mean_reward"] = np.mean(rewards)
            wandb.run.summary["top_reward"] = max(rewards)
            wandb.finish()

        if plotting:
            utils.plot(frame_idx, mean_rewards)
            utils.plot_test_trajectory(env, agent)
        return agent

    def set_agent(self, agent_str, **kwargs):

        self.agent = eval("ag."+agent_str)(self.state_dim, self.action_dim, **kwargs)


class observability(learning):
    def __init__(self, wandb_save=False, verbose=False, reward_type="PB", max_episodes=2000, max_steps=600, max_frames=1e5, max_epochs=20):

        super().__init__(wandb_save=wandb_save, verbose=verbose, reward_type=reward_type, max_episodes=max_episodes, max_steps=max_steps, max_frames=max_frames, max_epochs=max_epochs)
        self.state_dim = len(self.env.observation_space) * 2

    def learning_loop_online(self, agent_str, notebook=False, plotting=False):
        wandb.init() if self.wandb_save else None
        env = self.env

        if agent_str == "DuelDQN" or agent_str == "DuelDDQN":
            print("wrong agents, use Actor Critics: A2C or PPO")
            return

        agent = self.agent

        rewards = []
        mean_rewards = []
        frame_idx = 0

        for episodes in range(self.max_episodes):

            # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
            state = env.reset()
            # default velocity
            state_v = np.zeros(3)
            # markov state contains all the information
            state_mkv = np.hstack((state, state_v))

            episode_reward = 0
            I = 1

            for i in range(self.max_steps):

                # get action
                action = agent.get_action(state_mkv)

                # step through environment
                next_state, reward, done, = env.step(action)
                # add reward
                episode_reward += reward

                # update the agent with last experience
                next_state_mkv = self.get_augmented_state(state, next_state)
                loss = agent.online_update((state_mkv, action, reward, next_state_mkv, done), I)
                I *= agent.gamma

                wandb.log({'loss': loss}) if self.wandb_save else None

                # prepare for next iteration
                state = next_state
                state_mkv = next_state_mkv
                frame_idx += 1

                # if the episode is finished we stop there
                if done:
                    break

            # bookkeeping
            rewards.append(episode_reward)
            mean = np.mean(rewards[-50:])
            mean_rewards.append(mean)

            # we log or print depending on settings
            wandb.log({'episode_reward': episode_reward, "moving_average": mean}) if self.wandb_save else None
            print("Episode:", episodes, "|| Reward:", round(episode_reward),"|| Final State ", env._which_final_state().name) if self.verbose else None

            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(frame_idx, mean_rewards)
                if episodes % 500 == 0:
                    utils.plot_test_trajectory(env, agent)

            # if we spend a long time in the simulation
            if frame_idx > self.max_frames:
                break

        # log and show final trajectory
        if self.wandb_save:
            wandb.run.summary["mean_reward"] = np.mean(rewards)
            wandb.run.summary["top_reward"] = max(rewards)
            wandb.finish()

        if plotting:
            utils.plot(frame_idx, mean_rewards)
            utils.plot_test_trajectory(env, agent)
        return agent

    def learning_loop_offline(self, agent_str, buffer_size, batch_size, per_is, notebook=False, plotting=False, alpha=0.6, beta=0.4):

        wandb.init() if self.wandb_save else None
        env = self.env

        # initiate memory
        memory = utils.PER_IS_ReplayBuffer(buffer_size, alpha=alpha, state_dim=self.state_dim) if per_is else utils.ReplayBuffer(buffer_size)

        if agent_str != "DuelDQN" and agent_str != "DuelDDQN":
            print("wrong agents, use DQN agents: DuelDQN or DuelDDQN")
            return

        agent = self.agent

        rewards = []
        mean_rewards = []
        frame_idx = 0

        for episodes in range(self.max_episodes):

            # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
            state = env.reset()
            episode_reward = 0
            # default velocity
            state_v = np.zeros(3)
            # markov state contains all the information
            state_mkv = np.hstack((state, state_v))

            for i in range(self.max_steps):

                # get state
                action = agent.get_action(state_mkv)

                # step through environment
                next_state, reward, done, = env.step(action)
                next_state_mkv = self.get_augmented_state(state, next_state)
                # add reward
                episode_reward += reward

                memory.push(state_mkv, action, reward, next_state_mkv, done)

                # prepare for next iteration
                state = next_state
                state_mkv = next_state_mkv
                frame_idx += 1

                # if the episode is finished we stop there
                if done:
                    break

            # bookkeeping
            rewards.append(episode_reward)
            mean = np.mean(rewards[-50:])
            mean_rewards.append(mean)

            for epoch in range(self.max_epochs):
                if memory.__len__() > batch_size:
                    if per_is:
                        beta = 1 - (1-beta) * np.exp(-0.005 * episodes)  # we converge beta to 1
                        sample = memory.sample(batch_size, beta)
                        loss, tds = agent.update(
                            (sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']),
                            weights=sample['weights']
                        )
                        new_tds = np.abs(tds.cpu().numpy()) + 1e-6
                        memory.update_priorities(sample['indexes'], new_tds)
                    else:
                        sample = memory.sample(batch_size)
                        loss, _ = agent.update(sample)

                    wandb.log({'loss': loss}) if self.wandb_save else None

            # we log or print depending on settings
            wandb.log({'episode_reward': episode_reward, "moving_average": mean}) if self.wandb_save else None
            print("Episode:", episodes, "|| Reward:", round(episode_reward),"|| Final State ", env._which_final_state().name) if self.verbose else None

            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(frame_idx, mean_rewards)
                if plotting and episodes % 500 == 0:
                    utils.plot_test_trajectory(env, agent)

            # if we spend too long in the simulation
            if frame_idx > self.max_frames:
                break

        # log and show final trajectory
        if self.wandb_save:
            wandb.run.summary["mean_reward"] = np.mean(rewards)
            wandb.run.summary["top_reward"] = max(rewards)
            wandb.finish()

        if plotting:
            utils.plot(frame_idx, mean_rewards)
            utils.plot_test_trajectory(env, agent)
        return agent

    def get_augmented_state(self, state, next_state):
        """Returns the velocity of the next state"""
        return np.hstack((next_state, next_state-state))

if __name__=="__main__":
    # experiment = observability(verbose=True, max_episodes=3000)
    # experiment.set_agent("A2C", epsilon=0.01, lr=3e-4)
    # experiment.learning_loop_online("A2C", notebook=False, plotting=False)
    experiment = observability(max_frames=1e6, max_episodes=3000, verbose=True)
    experiment.set_agent("DuelDDQN", epsilon=0.0, lr=3e-4)
    experiment.learning_loop_offline("DuelDDQN", buffer_size=2 ** 14, batch_size=128, per_is=True, notebook=False,
                                     plotting=False)

