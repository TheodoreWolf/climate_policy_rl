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
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Learn:
    def __init__(self, wandb_save=False, verbose=False, reward_type="PB",
                 max_episodes=2000, max_steps=600, max_frames=1e5,
                 max_epochs=50, seed=0, gamma=0.99,
                 save_locally=False):

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
        self.save_locally = save_locally
        self.group_name = reward_type

        # run information in a dictionary
        self.data = {'rewards': [],
                     'moving_avg_rewards': [],
                     'moving_std_rewards': [],
                     'frame_idx': 0,
                     'episodes': 0,
                     'final_point': []
                     }

    def learning_loop_rollout(self, batch_size, buffer_size, notebook=False, plotting=False, config=None):
        """For PPO and A2C, these can't be updated fully offline"""
        # if str(self.agent) == "A2C":
        #     assert self.max_epochs == 1, "A2C is on-policy, can't update for more than one epoch, " \
        #                                  "max_epochs must be set to 1."
        #     assert batch_size == buffer_size, "A2C is on-policy, can't update in mini-batches."

        wandb.init(project="AYS_learning", entity="climate_policy_optim", config=config, job_type=str(self.agent),
                   group=self.group_name) \
            if self.wandb_save else None

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

                    episode_reward = 0
                    next_state = torch.Tensor(self.env.reset()).to(DEVICE)

                    # for notebook
                    if notebook and self.data['episodes'] % 10 == 0:
                        utils.plot(self.data)
                        if plotting and self.data['episodes'] % 500 == 0:
                            utils.plot_test_trajectory(self.env, self.agent)

                # get action and other stuff
                with torch.no_grad():
                    action, log_prob, entropy, value = self.agent.get_action_and_value(next_state)

                # step through environment
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())

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
                    pol_loss, val_loss = self.agent.update((obs[batch_idx], actions[batch_idx], values[batch_idx],
                                                            rewards[batch_idx], dones[batch_idx],
                                                            logprobs[batch_idx],
                                                            advantages[batch_idx], returns[batch_idx]))
                    wandb.log({"pol_loss": pol_loss, "val_loss": val_loss, "loss": val_loss + pol_loss}) \
                        if self.wandb_save else None

            # scheduler steps only if we have learnt something
            # if self.data['moving_avg_rewards'][-1] > 150:
            self.agent.critic_scheduler.step()
            self.agent.actor_scheduler.step()

            # if we spend a long time in the simulation
            if self.data['frame_idx'] > self.max_frames or self.data['episodes'] > self.max_episodes:
                break
        success_rate = self.data["final_point"].count("GREEN_FP")/self.data["episodes"]
        print(success_rate)
        # log data
        if self.wandb_save:
            wandb.run.summary["mean_reward"] = np.mean(self.data['rewards'])
            wandb.run.summary["top_reward"] = max(self.data['rewards'])
            wandb.run.summary["success_rate"] = success_rate
            wandb.run.summary["data"] = self.data
            wandb.finish()

        # show final trajectory
        if plotting:
            utils.plot(self.data)
            utils.plot_test_trajectory(self.env, self.agent)

        if self.save_locally:
            torch.save(self.agent.actor.state_dict(), "policy_net.pt")
            torch.save(self.agent.critic.state_dict(), "value_net.pt")
            np.save('run_data.npy', self.data)

    def learning_loop_offline(self, batch_size, buffer_size, per_is, notebook=False,
                              plotting=False, alpha=0.345, beta=0.236, config=None):
        """For DQN-based agents which can be updated offline which is more data efficient """

        wandb.init(project="AYS_learning", entity="climate_policy_optim", config=config, job_type=str(self.agent),
                   group=self.group_name) \
            if self.wandb_save else None
        # initiate memory
        self.memory = utils.PER_IS_ReplayBuffer(buffer_size, alpha=alpha, state_dim=self.state_dim) if per_is else utils.ReplayBuffer(buffer_size)

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
                # prepare for next iteration
                state = next_state
                self.data['frame_idx'] += 1

                # if the episode is finished we stop there
                if done:
                    break

            # bookkeeping
            self.append_data(episode_reward)

            # self.agent.scheduler.step()
            # for notebook
            if notebook and episodes % 10 == 0:
                utils.plot(self.data)
                if plotting and episodes % 200 == 0:
                    utils.plot_test_trajectory(self.env, self.agent)

            # if we spend too long in the simulation
            if self.data['frame_idx'] > self.max_frames:
                break
        success_rate = self.data["final_point"].count("GREEN_FP") / self.data["episodes"]
        print(success_rate)
        # log and show final trajectory
        if self.wandb_save:
            wandb.run.summary["mean_reward"] = np.mean(self.data['rewards'])
            wandb.run.summary["top_reward"] = max(self.data['rewards'])
            wandb.run.summary["success_rate"] = success_rate
            wandb.run.summary["data"] = self.data
            wandb.finish()

        if plotting:
            utils.plot(self.data)
            utils.plot_test_trajectory(self.env, self.agent)

    def set_agent(self, agent_str, pt_file_path=None, second_path=None, **kwargs):
        """Set the agent to the environment with specific parameters or weights"""
        self.agent = eval("ag." + agent_str)(self.state_dim, self.action_dim, gamma=self.gamma, **kwargs)
        if pt_file_path is not None:
            if agent_str == "PPO" or agent_str == "A2C":
                self.agent.actor.load_state_dict(torch.load(pt_file_path))
                self.agent.critic.load_state_dict(torch.load(second_path))
            else:
                self.agent.policy_net.load_state_dict(torch.load(pt_file_path))
                self.agent.target_net.load_state_dict(torch.load(pt_file_path))

    def append_data(self, episode_reward):
        """We append the latest episode reward and calculate moving averages and moving standard deviations"""

        self.data['rewards'].append(episode_reward)
        self.data['moving_avg_rewards'].append(np.mean(self.data['rewards'][-50:]))
        self.data['moving_std_rewards'].append(np.std(self.data['rewards'][-50:]))
        self.data['episodes'] += 1
        self.data['final_point'].append(self.env.which_final_state().name)

        # we log or print depending on settings
        wandb.log({'episode_reward': episode_reward,
                   "moving_average": self.data['moving_avg_rewards'][-1]}) \
            if self.wandb_save else None

        print("Episode:", self.data['episodes'], "|| Reward:", round(episode_reward), "|| Final State ",
              self.env.which_final_state().name) \
            if self.verbose else None

    def test_agent(self, n_points=100, max_steps=10000, s_default=0.5) -> plt:
        """Test the agent on different initial conditions to see if it can escape"""
        grid_size = int(np.sqrt(n_points))
        results = np.zeros((n_points, 1))
        test_states = np.zeros((n_points, self.state_dim))

        for a in range(grid_size):
            for y in range(grid_size):
                test_states[a * grid_size + y] = \
                    np.array([0.45 + a * 1 / (grid_size * 10), 0.45 + y * 1 / (grid_size * 10), s_default])

        for i in range(len(test_states)):
            state = self.env.reset_for_state(test_states[i])
            for steps in range(max_steps):
                action = self.agent.get_action(state, testing=True)
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    results[i] = int(self.env.which_final_state().value)
                    break
                state = next_state
        utils.plot_end_state_matrix(results)

    def initialisation_values(self, n_points=100, s_default=0.5) -> plt:
        """Create a plot of the state values at different initialisations"""
        grid_size = int(np.sqrt(n_points))
        test_states = np.zeros((n_points, self.state_dim))
        for a in range(grid_size):
            for y in range(grid_size):
                test_states[a * grid_size + y] = \
                    np.array([0.45 + a * 1 / (grid_size * 10), 0.45 + y * 1 / (grid_size * 10), s_default])
        if str(self.agent) == "A2C" or str(self.agent) == "PPO":
            results = self.agent.critic(torch.from_numpy(test_states).float().to(DEVICE))
        else:
            results = torch.max((self.agent.target_net(torch.from_numpy(test_states).float().to(DEVICE))), dim=1)[0]

        plt.imshow(results.view(grid_size, grid_size).detach().cpu(), extent=(0.45, 0.55, 0.55, 0.45))
        plt.ylabel("Y")
        plt.xlabel("A")
        plt.colorbar()

    def initialisation_actions(self, n_points=100, s_default=0.5) -> plt:
        """Make a plot of the best action to do at initialisation"""
        grid_size = int(np.sqrt(n_points))
        test_states = np.zeros((n_points, self.state_dim))
        for a in range(grid_size):
            for y in range(grid_size):
                test_states[a * grid_size + y] = \
                    np.array([0.45 + a * 1 / (grid_size * 10), 0.45 + y * 1 / (grid_size * 10), s_default])
        if str(self.agent) == "A2C" or str(self.agent) == "PPO":
            results = torch.argmax(self.agent.actor(torch.from_numpy(test_states).float().to(DEVICE)), dim=1)
        else:
            results = torch.argmax((self.agent.target_net(torch.from_numpy(test_states).float().to(DEVICE))), dim=1)
        utils.plot_action_matrix(results.detach().cpu().numpy())

    def feature_plots(self, samples, buffer=None, v=False, actor=False) -> plt:
        """To make feature importance plots"""
        if buffer is None:
            self.sample_states(samples)
        else:
            self.samples = buffer
        if str(self.agent) == "A2C" or str(self.agent) == "PPO":
            if actor:
                agent_net = self.agent.actor
                utils.feature_importance(agent_net, self.samples, samples, v, scalar=False)
            else:
                agent_net = self.agent.critic
                utils.feature_importance(agent_net, self.samples, samples, v, scalar=True)
        else:
            agent_net = self.agent.target_net
            utils.feature_importance(agent_net, self.samples, samples, v)

    def plot_trajectory(self, start_state=None, steps=600, fname=None) -> plt:
        utils.plot_test_trajectory(self.env, self.agent, max_steps=steps, test_state=start_state, fname=fname)

    def sample_states(self, samples):
        """Sample states from the environment"""
        self.samples = utils.ReplayBuffer(samples)
        while len(self.samples) < samples:
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.get_action(state, testing=True)
                next_state, reward, done, _ = self.env.step(action)
                self.samples.push(state, action, reward, next_state, done)
                state = next_state

        return self.samples

if __name__ == "__main__":
    a=1
    # experiment = Learning(max_frames=5e5, verbose=True, max_epochs=1, seed=2, reward_type='PB', max_episodes=20000,
    #                       save_locally=True)
    # experiment.set_agent("A2C", epsilon=0.002, lamda=0.81, lr_critic=0.004, lr_actor=0.0013, max_grad_norm=100,
    #                      actor_decay=1., critic_decay=1.)
    # experiment.learning_loop_rollout(128, 128, plotting=False)

    # experiment = Learning(max_frames=5e5, gamma=0.894, verbose=True, max_epochs=1, seed=0, reward_type='PB', max_episodes=20000,
    #                       save_locally=False)
    # experiment.set_agent("A2C", epsilon=0.0241, lamda=0.161, lr_critic=0.005984, lr_actor=0.00988946, max_grad_norm=1000,
    #                      actor_decay=0.9968, critic_decay=0.99755)
    # experiment.learning_loop_rollout(128, 128, plotting=True)
    experiment = Learn(max_frames=5e5, gamma=0.99, verbose=True, max_epochs=1, seed=1, reward_type='PB',
                       max_episodes=20000,
                       save_locally=False)
    experiment.set_agent("A2C", epsilon=0.2758, lamda=0.99, lr_critic=0.0003, lr_actor=0.0003, max_grad_norm=1000,
                         actor_decay=1.0, critic_decay=1.0)
    experiment.learning_loop_rollout(64, 64, plotting=True)