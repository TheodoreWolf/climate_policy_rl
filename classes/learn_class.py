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


class Learning:
    def __init__(self, wandb_save=False, verbose=False, reward_type="PB",
                 max_episodes=2000, max_steps=600, max_frames=1e5,
                 max_epochs=50, seed=0, gamma=0.99):

        # environment
        self.env = AYS_Environment(reward_type=reward_type, discount=gamma)
        self.state_dim = len(self.env.observation_space)
        self.action_dim = len(self.env.action_space)
        self.gamma = gamma

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

        # run information in a dictionary
        self.data = {'rewards': [],
                     'moving_avg_rewards': [],
                     'moving_std_rewards': [],
                     'frame_idx': 0,
                     'episodes': 0,
                     'final_point': []
                     }

    def learning_loop_rollout(self, batch_size, buffer_size, notebook=False, plotting=False):
        """For PPO and A2C, these can't be updated fully offline"""

        if str(self.agent) == "A2C":
            assert self.max_epochs == 1, "A2C is on-policy, can't update for more than one epoch, " \
                                         "max_epochs must be set to 1."
            assert batch_size == buffer_size, "A2C is on-policy, can't update in mini-batches."

        wandb.init() if self.wandb_save else None

        # initialise recursive variables
        episode_reward = 0
        next_state = torch.Tensor(self.env.reset()).to(DEVICE)
        next_done = torch.zeros(1).to(DEVICE)

        # the loop is separated in two parts: number of updates and size of batch
        for update in range(int(self.max_frames // buffer_size)):

            # reset data arrays
            obs = torch.zeros((buffer_size, self.state_dim)).to(DEVICE)
            actions = torch.zeros(buffer_size).to(DEVICE)
            logprobs = torch.zeros(buffer_size).to(DEVICE)
            rewards = torch.zeros(buffer_size).to(DEVICE)
            dones = torch.zeros(buffer_size).to(DEVICE)
            values = torch.zeros(buffer_size).to(DEVICE)

            for i in range(buffer_size):
                # update frame (global step)
                self.data['frame_idx'] += 1
                # append batch data
                dones[i] = next_done
                obs[i] = next_state

                if next_done:
                    # append data
                    self.append_data(episode_reward)

                    print("Episode:", self.data['episodes'], "|| Reward:", round(episode_reward),
                          "|| Final_state", self.env.which_final_state().name) if self.verbose else None

                    wandb.log({'episode_reward': episode_reward,
                               "moving_average": self.data['moving_avg_reward'][-1]}) if self.wandb_save else None

                    episode_reward = 0
                    next_state = torch.Tensor(self.env.reset()).to(DEVICE)

                    # for notebook
                    if notebook and self.data['episodes'] % 10 == 0:
                        utils.plot(self.data)
                        if self.data['episodes'] % 500 == 0:
                            utils.plot_test_trajectory(self.env, self.agent)

                # get action and other stuff
                with torch.no_grad():
                    action, log_prob, entropy, value = self.agent.get_action_and_value(next_state)

                # step through environment
                next_state, reward, done, _ = self.env.step(action)

                # append batch data
                actions[i] = action
                logprobs[i] = log_prob
                rewards[i] = reward
                values[i] = value
                next_state, next_done = torch.Tensor(next_state).to(DEVICE), torch.Tensor([done]).to(DEVICE)

                # update episode reward
                episode_reward += reward

            # get the value
            with torch.no_grad():
                next_value = self.agent.critic(next_state)

            # compute advantages (in buffer order)
            returns, advantages = self.agent.compute_gae(values, dones, rewards, next_value, next_done)
            buffer_idx = np.arange(buffer_size)

            # randomise the experience to un-correlate
            for epochs in range(self.max_epochs):
                np.random.shuffle(buffer_idx)
                for start in range(0, buffer_size, batch_size):
                    end = start + batch_size
                    batch_idx = buffer_idx[start:end]
                    self.agent.update((obs[batch_idx], actions[batch_idx], values[batch_idx],
                                       rewards[batch_idx], dones[batch_idx],
                                       logprobs[batch_idx],
                                       advantages[batch_idx], returns[batch_idx]))
            # scheduler steps only if we have learnt something
            # if self.data['moving_avg_rewards'][-1] > 150:
            self.agent.critic_scheduler.step()
            self.agent.actor_scheduler.step()

            # if we spend a long time in the simulation
            if self.data['frame_idx'] > self.max_frames or self.data['episodes'] > self.max_episodes:
                break

        # log data
        if self.wandb_save:
            wandb.run.summary["mean_reward"] = np.mean(self.data['rewards'])
            wandb.run.summary["top_reward"] = max(self.data['rewards'])
            wandb.finish()

        # show final trajectory
        if plotting:
            utils.plot(self.data)
            utils.plot_test_trajectory(self.env, self.agent)

    def learning_loop_offline(self, batch_size, buffer_size, per_is, notebook=False,
                              plotting=False, alpha=0.4, beta=0.2):
        """For DQN-based agents which can be updated offline which is more data efficient """

        wandb.init() if self.wandb_save else None

        # initiate memory
        self.memory = utils.PER_IS_ReplayBuffer(buffer_size, alpha=alpha) if per_is else utils.ReplayBuffer(buffer_size)

        for episodes in range(self.max_episodes):

            # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
            state = self.env.reset()
            episode_reward = 0

            for i in range(self.max_steps):

                # get state
                action = self.agent.get_action(state)

                # step through environment
                next_state, reward, done, _ = self.env.step(action)

                # add reward
                episode_reward += reward

                self.memory.push(state, action, reward, next_state, done)

                # prepare for next iteration
                state = next_state
                self.data['frame_idx'] += 1

                # if the episode is finished we stop there
                if done:
                    break

            # bookkeeping
            self.append_data(episode_reward)

            # we loop through epochs
            for epoch in range(self.max_epochs):
                # once the buffer is large enough for a batch
                if len(self.memory) > batch_size:
                    # if we are using prioritised experience replay buffer with importance sampling
                    if per_is:
                        beta = 1 - (1 - beta) * np.exp(-0.005 * episodes)  # we converge beta to 1
                        sample = self.memory.sample(batch_size, beta)
                        loss, tds = self.agent.update(
                            (sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']),
                            weights=sample['weights']
                        )
                        new_tds = np.abs(tds.cpu().numpy()) + 1e-6
                        self.memory.update_priorities(sample['indexes'], new_tds)
                    # otherwise we just uniformly sample
                    else:
                        sample = self.memory.sample(batch_size)
                        loss, _ = self.agent.update(sample)

                    wandb.log({'loss': loss}) if self.wandb_save else None

            # we log or print depending on settings
            wandb.log({'episode_reward': episode_reward,
                       "moving_average": self.data['moving_avg_reward']}) if self.wandb_save else None

            print("Episode:", episodes, "|| Reward:", round(episode_reward), "|| Final State ",
                  self.env.which_final_state().name) if self.verbose else None

            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(self.data)
                if plotting and episodes % 200 == 0:
                    utils.plot_test_trajectory(self.env, self.agent)

            # if we spend too long in the simulation
            if self.data['frame_idx'] > self.max_frames:
                break

        # log and show final trajectory
        if self.wandb_save:
            wandb.run.summary["mean_reward"] = np.mean(self.data['rewards'])
            wandb.run.summary["top_reward"] = max(self.data['rewards'])
            wandb.finish()

        if plotting:
            utils.plot(self.data)
            utils.plot_test_trajectory(self.env, self.agent)

    def set_agent(self, agent_str, **kwargs):
        """Set the agent to the environment with specific parameters"""
        self.agent = eval("ag." + agent_str)(self.state_dim, self.action_dim, gamma=self.gamma, **kwargs)

    def append_data(self, episode_reward):
        """We append the latest episode reward and calculate moving averages and moving standard deviations"""

        self.data['rewards'].append(episode_reward)
        self.data['moving_avg_rewards'].append(np.mean(self.data['rewards'][-50:]))
        self.data['moving_std_rewards'].append(np.std(self.data['rewards'][-50:]))
        self.data['episodes'] += 1
        self.data['final_point'].append(self.env.which_final_state().name)

    def test_agent(self, n_points=100, max_steps=10000, state_size=3):
        """Test the agent on different initial conditions"""
        grid_size = int(np.sqrt(n_points))
        results = np.zeros((n_points, 1))
        test_states = np.zeros((n_points, state_size))

        for a in range(grid_size):
            for y in range(grid_size):
                test_states[a*grid_size + y] = np.array([0.45 + a * 1/(grid_size*10), 0.45 + y * 1/(grid_size*10), 0.5])

        for i in range(len(test_states)):
            state = self.env.reset_for_state(test_states[i])
            for steps in range(max_steps):
                action = self.agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    results[i] = int(self.env.which_final_state().value)
                    break
                state = next_state
        utils.plot_matrix(results)

    def feature_plots(self, samples):
        """To make feature importance plots"""
        self.samples = utils.ReplayBuffer(samples)
        while len(self.samples) < samples:
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.samples.push(state, action, reward, next_state, done)
                state = next_state
        if str(self.agent)=="A2C" or str(self.agent)=="PPO":
            agent_net = self.agent.actor
        else:
            agent_net = self.agent.target_net
        utils.feature_importance(agent_net, self.samples, samples)

    def plot_trajectory(self, start_state, steps=600):
        utils.plot_test_trajectory(self.env, self.agent, max_steps=steps, test_state=start_state)



    # def learning_loop_online(self, agent_str, notebook=False, plotting=False):
    #
    #     wandb.init() if self.wandb_save else None
    #     env = self.env
    #
    #     if agent_str == "DuelDQN" or agent_str == "DuelDDQN":
    #         print("wrong agents, use Actor Critics: A2C or PPO")
    #         return
    #
    #     agent = self.agent
    #
    #     rewards = []
    #     mean_rewards = []
    #     std_rewards = []
    #     frame_idx = 0
    #
    #     for episodes in range(self.max_episodes):
    #
    #         # reset environment to a random state (0.5, 0.5, 0.5) * gaussian noise
    #         state = env.reset()
    #         episode_reward = 0
    #         I = 1
    #
    #         for i in range(self.max_steps):
    #
    #             # get state
    #             action = agent.get_action(state)
    #
    #             # step through environment
    #             next_state, reward, done, = env.step(action)
    #
    #             # add reward
    #             episode_reward += reward
    #
    #             # update the agent with last experience
    #             loss = agent.online_update((state, action, reward, next_state, done), I)
    #             I *= agent.gamma
    #
    #             wandb.log({'loss': loss}) if self.wandb_save else None
    #
    #             # prepare for next iteration
    #             state = next_state
    #             frame_idx += 1
    #
    #             # if the episode is finished we stop there
    #             if done:
    #                 break
    #
    #         # bookkeeping
    #         rewards.append(episode_reward)
    #         mean = np.mean(rewards[-50:])
    #         std = np.std(rewards[-50:])
    #         mean_rewards.append(mean)
    #         std_rewards.append(std)
    #
    #         # we log or print depending on settings
    #         wandb.log({'episode_reward': episode_reward, "moving_average": mean}) if self.wandb_save else None
    #         print("Episode:", episodes, "|| Reward:", round(episode_reward),
    #               "|| Final State ", env.which_final_state().name) if self.verbose else None
    #
    #         # for notebook
    #         if notebook and episodes % 10 == 0:
    #             utils.plot(frame_idx, mean_rewards, std_rewards)
    #             if episodes % 500 == 0:
    #                 utils.plot_test_trajectory(env, agent)
    #
    #         # if we spend a long time in the simulation
    #         if frame_idx > self.max_frames:
    #             break
    #
    #     # log and show final trajectory
    #     if self.wandb_save:
    #         wandb.run.summary["mean_reward"] = np.mean(rewards)
    #         wandb.run.summary["top_reward"] = max(rewards)
    #         wandb.finish()
    #
    #     if plotting:
    #         utils.plot(frame_idx, mean_rewards, std_rewards)
    #         utils.plot_test_trajectory(env, agent)
    #     return rewards



if __name__ == "__main__":
    # experiment = Learning(max_frames=1e3, verbose=True, max_epochs=75, seed=0, reward_type='PB')
    # experiment.set_agent("DuelDDQN")
    # experiment.learning_loop_offline(64, 10000, per_is=True, plotting=False)
    # experiment.test_agent(n_points=100)
    # experiment.feature_plots(100)
    experiment = Learning(max_frames=1e6, verbose=True, max_epochs=1, seed=1, reward_type='PB', max_episodes=40000)
    experiment.set_agent("A2C", epsilon=0.27, lamda=0.81, lr_critic=0.004, lr_actor=0.0013, max_grad_norm=100,
                         actor_decay=1., critic_decay=1.)
    experiment.learning_loop_rollout(128, 128, plotting=True)
