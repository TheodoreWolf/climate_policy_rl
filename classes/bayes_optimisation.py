from learning_loop import learning_loop
from bayes_opt import BayesianOptimization


def optimise_agent():
    """Optimise hyperparameters for agent and replay buffer"""

    def agent_evaluation(batch_size, random_exp, lr, buffer_size, discount, beta):
        """We evaluate the agent for different hyperparameter settings"""

        buffer_size = round(buffer_size)
        random_exp = round(random_exp)
        batch_size = round(batch_size)

        return learning_loop(AGENT=AGENT,
                             UPDATE_ONLINE=False,
                             BATCH_SIZE=2**batch_size,
                             MAX_EPISODES=2000,
                             RANDOM_EXPERIENCE=10**random_exp,
                             LEARNING_RATE=10**(-lr),
                             BUFFER_SIZE=2 ** buffer_size,
                             REWARD_TYPE='PB',
                             DT=1,
                             SCHEDULER=(False, 1000, 0.5),
                             SEED=0,
                             MAX_STEPS=600,
                             DISCOUNT=discount,
                             MAX_FRAMES=5e4,
                             NAME=None,
                             PER_IS=False,
                             PLOT=False,
                             JOB_TYPE="hyperparam_"+AGENT,
                             BETA=beta
                             )

    optimiser = BayesianOptimization(f=agent_evaluation,
                                     pbounds={'batch_size': (4, 10), 'random_exp': (0, 3), 'buffer_size': (10, 15), 'lr': (2, 5), 'discount': (0.5, 1), 'beta': (0, 1)},
                                     verbose=2)
    optimiser.maximize(n_iter=30, init_points=5)
    print('final result', optimiser.max)


if __name__ == "__main__":
    AGENT= "A2C"
    optimise_agent()
