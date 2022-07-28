try:
    from Envs.AYS.AYS_Environment import AYS_Environment
    import Learning.agents as ag
    import Learning.utils as utils
except:
    from .Envs.AYS.AYS_Environment import AYS_Environment
    from .Learning import agents as ag
    from .Learning import utils as utils
import random
import numpy as np
import torch
import wandb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class learning:
    def __init__(self, wandb_save=False, verbose=False, reward_type="PB",
                 max_episodes=2000, max_steps=600, max_frames=1e5,
                 max_epochs=50, seed=0, gamma=0.99, labda=0.95):

        # environment
        self.env = AYS_Environment(reward_type=reward_type, discount=gamma)
        self.state_dim = len(self.env.observation_space)
        self.action_dim = len(self.env.action_space)

        # seeds
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

        # computing returns
        self.gamma = gamma
        self.labda = labda

    def learning_loop_rollout(self, batch_size, buffer_size, notebook=False, plotting=False):
        # we implement rollout updates for PPO
        wandb.init() if self.wandb_save else None
        env = self.env

        agent = self.agent

        ep_rewards = []
        mean_rewards = []
        std_rewards = []
        frame_idx = 0
        episode_reward = 0
        episodes = 0
        next_state = torch.Tensor(env.reset()).to(DEVICE)
        next_done = torch.zeros(1).to(DEVICE)

        for update in range(int(self.max_frames//buffer_size)):

            obs = torch.zeros((buffer_size, self.state_dim)).to(DEVICE)
            actions = torch.zeros(buffer_size).to(DEVICE)
            logprobs = torch.zeros(buffer_size).to(DEVICE)
            rewards = torch.zeros(buffer_size).to(DEVICE)
            dones = torch.zeros(buffer_size).to(DEVICE)
            values = torch.zeros(buffer_size).to(DEVICE)

            for i in range(buffer_size):

                frame_idx += 1
                dones[i] = next_done
                obs[i] = next_state
                if next_done:
                    episodes += 1
                    print("Episode:", episodes, "|| Reward:", round(episode_reward),
                          "|| Final_state", env._which_final_state().name) if self.verbose else None
                    # we log or print depending on settings
                    # bookkeeping
                    ep_rewards.append(episode_reward)
                    mean = np.mean(ep_rewards[-50:])
                    std = np.std(ep_rewards[-50:])
                    mean_rewards.append(mean)
                    std_rewards.append(std)
                    wandb.log({'episode_reward': episode_reward, "moving_average": mean}) if self.wandb_save else None

                    episode_reward = 0
                    next_state = torch.Tensor(env.reset()).to(DEVICE)


                # get action and other stuff
                with torch.no_grad():
                    action, log_prob, entropy, value = agent.get_action_and_value(next_state)

                # step through environment
                next_state, reward, done, _ = env.step(action)

                actions[i] = action
                logprobs[i] = log_prob
                rewards[i] = reward
                values[i] = value
                next_state, next_done = torch.Tensor(next_state).to(DEVICE), torch.Tensor([done]).to(DEVICE)
                episode_reward += reward
            with torch.no_grad():
                next_value = agent.critic(next_state)
            returns, advantages = self.compute_gae(values, dones, rewards, next_value, next_done)
            buffer_idx = np.arange(buffer_size)

            for epochs in range(self.max_epochs):
                np.random.shuffle(buffer_idx)
                for start in range(0, buffer_size, batch_size):
                    end = start + batch_size
                    batch_idx = buffer_idx[start:end]
                    agent.update((obs[batch_idx], actions[batch_idx],values[batch_idx],
                                 rewards[batch_idx], dones[batch_idx],
                                  logprobs[batch_idx],
                                  advantages[batch_idx], returns[batch_idx]))

            agent.critic_scheduler.step()
            agent.actor_scheduler.step()

            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(frame_idx, mean_rewards, std_rewards)
                if episodes % 500 == 0:
                    utils.plot_test_trajectory(env, agent)

            # if we spend a long time in the simulation
            if frame_idx > self.max_frames:
                break

        # log and show final trajectory
        if self.wandb_save:
            wandb.run.summary["mean_reward"] = np.mean(ep_rewards)
            wandb.run.summary["top_reward"] = max(ep_rewards)
            wandb.finish()

        if plotting:
            utils.plot(frame_idx, mean_rewards, std_rewards)
            utils.plot_test_trajectory(env, agent)
        return ep_rewards

    def learning_loop_online(self, agent_str, notebook=False, plotting=False):

        wandb.init() if self.wandb_save else None
        env = self.env

        if agent_str == "DuelDQN" or agent_str == "DuelDDQN":
            print("wrong agents, use Actor Critics: A2C or PPO")
            return

        agent = self.agent

        rewards = []
        mean_rewards = []
        std_rewards = []
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
            std = np.std(rewards[-50:])
            mean_rewards.append(mean)
            std_rewards.append(std)

            # we log or print depending on settings
            wandb.log({'episode_reward': episode_reward, "moving_average": mean}) if self.wandb_save else None
            print("Episode:", episodes, "|| Reward:", round(episode_reward),
                  "|| Final State ", env._which_final_state().name) if self.verbose else None

            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(frame_idx, mean_rewards, std_rewards)
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
            utils.plot(frame_idx, mean_rewards, std_rewards)
            utils.plot_test_trajectory(env, agent)
        return rewards

    def learning_loop_offline(self, agent_str, buffer_size, batch_size, per_is, notebook=False,
                              plotting=False, alpha=0.6, beta=0.4):

        wandb.init() if self.wandb_save else None
        env = self.env

        # initiate memory
        self.memory = utils.PER_IS_ReplayBuffer(buffer_size, alpha=alpha) if per_is else utils.ReplayBuffer(buffer_size)

        if agent_str != "DuelDQN" and agent_str != "DuelDDQN":
            print("wrong agents, use DQN agents: DuelDQN or DuelDDQN")
            return

        agent = self.agent

        rewards = []
        mean_rewards = []
        std_rewards = []
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

                self.memory.push(state, action, reward, next_state, done)

                # prepare for next iteration
                state = next_state
                frame_idx += 1

                # if the episode is finished we stop there
                if done:
                    break

            # bookkeeping
            rewards.append(episode_reward)

            mean = np.mean(rewards[-50:])
            std = np.std(rewards[-50:])

            mean_rewards.append(mean)
            std_rewards.append(std)

            for epoch in range(self.max_epochs):
                if self.memory.__len__() > batch_size:
                    if per_is:
                        beta = 1 - (1-beta) * np.exp(-0.005 * episodes)  # we converge beta to 1
                        sample = self.memory.sample(batch_size, beta)
                        loss, tds = agent.update(
                            (sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']),
                            weights=sample['weights']
                        )
                        new_tds = np.abs(tds.cpu().numpy()) + 1e-6
                        self.memory.update_priorities(sample['indexes'], new_tds)
                    else:
                        sample = self.memory.sample(batch_size)
                        loss, _ = agent.update(sample)

                    wandb.log({'loss': loss}) if self.wandb_save else None
            # we log or print depending on settings
            wandb.log({'episode_reward': episode_reward, "moving_average": mean}) if self.wandb_save else None
            print("Episode:", episodes, "|| Reward:", round(episode_reward),"|| Final State ", env._which_final_state().name) if self.verbose else None

            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(frame_idx, mean_rewards, std_rewards)
                if plotting and episodes % 200 == 0:
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
            utils.plot(frame_idx, mean_rewards, std_rewards)
            utils.plot_test_trajectory(env, agent)
        return rewards

    def set_agent(self, agent_str, **kwargs):

        self.agent = eval("ag."+agent_str)(self.state_dim, self.action_dim, **kwargs)

    def compute_gae(self, values, dones, rewards, next_value, next_done):
        last_adv = 0
        buffer_size = len(rewards)
        advantages = torch.zeros_like(rewards).to(DEVICE)
        for t in reversed(range(buffer_size)):
            if t == buffer_size - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = last_adv = delta + self.gamma * self.labda * nextnonterminal * last_adv
        returns = advantages + values
        return returns, advantages

class MarkovState(learning):
    def __init__(self, wandb_save=False, verbose=False, reward_type="PB", max_episodes=2000, max_steps=600, max_frames=1e5, max_epochs=50):
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
        std_rewards = []
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
                next_state_mkv = self.get_diff_state(state, state)
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
            std = np.std(rewards[-50:])
            mean = np.mean(rewards[-50:])
            mean_rewards.append(mean)
            std_rewards.append(std)

            # we log or print depending on settings
            wandb.log({'episode_reward': episode_reward, "moving_average": mean}) if self.wandb_save else None
            print("Episode:", episodes, "|| Reward:", round(episode_reward),"|| Final State ", env._which_final_state().name) if self.verbose else None

            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(frame_idx, mean_rewards, std_rewards)
                if plotting and episodes % 500 == 0:
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
            utils.plot(frame_idx, mean_rewards, std_rewards)
            utils.plot_test_trajectory(env, agent)
        return rewards

    def learning_loop_offline(self, agent_str, buffer_size, batch_size, per_is, notebook=False, plotting=False, alpha=0.6, beta=0.4):

        wandb.init() if self.wandb_save else None
        env = self.env

        # initiate memory
        self.memory = utils.PER_IS_ReplayBuffer(buffer_size, alpha=alpha, state_dim=self.state_dim) if per_is else utils.ReplayBuffer(buffer_size)

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

                self.memory.push(state_mkv, action, reward, next_state_mkv, done)

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
                if self.memory.__len__() > batch_size:
                    if per_is:
                        beta = 1 - (1-beta) * np.exp(-0.005 * episodes)  # we converge beta to 1
                        sample = self.memory.sample(batch_size, beta)
                        loss, tds = agent.update(
                            (sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']),
                            weights=sample['weights']
                        )
                        new_tds = np.abs(tds.cpu().numpy()) + 1e-6
                        self.memory.update_priorities(sample['indexes'], new_tds)
                    else:
                        sample = self.memory.sample(batch_size)
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
        return rewards

    def get_diff_state(self, state, next_state):
        """Returns the velocity of the next state"""
        return np.hstack((next_state, next_state-state))

    def get_velocity_state(self, next_state, action):
        A, Y, S = next_state

        sigma = [4e12, 4e12, 2.83e12, 2.83e12]
        beta = [0.03, 0.015, 0.03, 0.015]
        gamma = 1/(1+(S/sigma[action])**2)
        phi = 4.7e10
        U = Y/147
        R = (1-gamma)*U
        E = (U-R)/phi

        dA = E - A/50
        dY = beta[action]*Y - 8.57e-5*A
        dS = R - S/50

        v = np.array([dA, dY, dS])
        return np.hstack((next_state, v))

if __name__=="__main__":
    # experiment = MarkovState(max_frames=1e6, max_episodes=3000, verbose=True)
    # experiment.set_agent("PPOsplit", epsilon=0.1, lr=3e-4)
    # experiment.learning_loop_offline("PPOsplit", buffer_size=2 ** 14, batch_size=128, per_is=True, notebook=False,
    #                                  plotting=False)
    experiment = learning(max_frames=1e5, verbose=True, max_epochs=10)
    experiment.set_agent("PPOsplit", epsilon=0.0, max_grad_norm=1000, )
    experiment.learning_loop_rollout(64, 640)
