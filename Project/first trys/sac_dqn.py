import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from gym import Env
from collections import OrderedDict


class DynEnv(Env):

    def __init__(self, reward_type=None):
        ics = dict(L=2480.,
                   A=758.0,
                   G=1125,
                   T=5.053333333333333e-6,
                   P=6e9,
                   K=6e13,
                   S=5e11
                   )
        pars = dict(Sigma=1.5 * 1e8,
                    Cstar=5500,
                    a0=0.03,
                    aT=3.2 * 1e3,
                    l0=26.4,
                    lT=1.1 * 1e6,
                    delta=0.01,
                    m=1.5,
                    g=0.02,
                    p=0.04,
                    Wp=2000,
                    q0=20,
                    b=5.4 * 1e-7,
                    yE=147,
                    eB=4 * 1e10,
                    eF=4 * 1e10,
                    i=0.25,
                    wL=0,
                    k0=0.1,
                    aY=0.,
                    aB=3e5,
                    aF=5e6,
                    aR=7e-18,
                    sS=1. / 50.,
                    sR=1.,
                    ren_sub=.5,
                    carbon_tax=.5,
                    i_DG=0.1,
                    L0=0,
                    )
        self.action_space = [(False, False, False),
                        (True, False, False), (False, True, False), (False, False, True),
                        (True, True, False), (True, False, True), (False, True, True),
                        (True, True, True)]

        self.carbon_tax = 0.5
        self.ren_sub = 0.5
        # Initialise class variables
        self.dt = 1
        self.params = pars
        self.initials = ics
        self.action_dim = 8
        self.observation_space = np.zeros(7)

        # We initialise the environment with a reset
        _ = self.reset()

        # Get the reward type
        self.reward_type = reward_type
        self.reward_function = self.get_reward_function(reward_type)

        # Planetary boundaries
        self.A_PB = 945/758  # TODO check these values
        self.W_PB = 1.84e4/18588
        self.P_PB = 1e6/6e9
        self.PB_vec = np.array([self.A_PB, self.W_PB, self.P_PB])

    def step(self, action_t) -> (np.array, float, bool):
        """The main step function where we call the others from"""

        next_t = self.t + self.dt

        # we adjust the parameters given the actions taken
        self.adjust_parameters(action_t)

        # we call the ode solver with the new parameters
        self.state = self.ode_solver(next_t)

        # Check if we have reached a final state
        if self.state_is_final():
            self.final_state = True

        reward = self.reward_function()
        self.t = next_t

        return self.normalise(self.state), reward, self.final_state

    def reset(self):
        """Resetting the environment at the start of an episode"""

        self.state = np.array(list(self.initials.values()))
        self.t = 0
        self.final_state = False
        self.adjust_parameters()

        return self.state

    def ode_solver(self, next_t) -> np.array:
        """This is where we solve the dynamical system of equations to get the next state"""

        # we use the scipy ode solver: odeint
        ode_solutions = odeint(func=self.dynamic_eqs,
                               y0=self.state,
                               t=[self.t, next_t],
                               mxstep=50000)

        # we take the values of the variables on the last time step as our next state
        return ode_solutions[-1, :]

    def dynamic_eqs(self, LAGTPKS, t) -> list:
        """The differential equations we have to solve for the next step"""

        # We define variables for simpler notation
        def phot(A, T):
            """Photosynthesis"""
            return (self.params['l0'] - self.params['lT'] * T) * np.sqrt(A / self.params['Sigma'])

        def resp(T):
            """Respiration"""
            return self.params['a0'] + self.params['aT'] * T

        def diff(L, A, G=0.):
            """Diffusion between atmosphere and ocean"""
            return self.params['delta'] * (self.params['Cstar'] - L - G - (1 + self.params['m']) * A)

        def fert(W):
            """Human fertility"""
            return 2 * self.params['p'] * self.params['Wp'] * W / (self.params['Wp'] ** 2 + W ** 2)

        def mort(W):
            """Human mortality"""
            return self.params['q0'] / (W +0.0001)

        # We unpack the state variables and derived variables
        L, A, G, T, P, K, S = LAGTPKS
        B, F, R, Y, W = self.compute_derived_variables()

        # clamp here
        L = np.amin([np.amax([L, 1e-12]), self.params['Cstar']])
        A = np.amin([np.amax([A, 1e-12]), self.params['Cstar']])
        G = np.amin([np.amax([G, 1e-12]), self.params['Cstar']])
        T = np.amax([T, 1e-12])
        P = np.amax([P, 1e-12])
        K = np.amax([K, 1e-12])
        S = np.amax([S, 1e-12])

        # odes to solve
        dL = (phot(A, T) - resp(T)) * L - B
        dA = -dL + diff(L, A, G)
        dG = -F
        dT = self.params['g'] * (A / self.params['Sigma'] - T)
        dP = P * (fert(W) - mort(W))
        dK = self.params['i'] * Y - self.params['k0'] * K
        dS = self.params['sR'] * R - self.params['sS'] * S

        return [dL, dA, dG, dT, dP, dK, dS]

    def compute_derived_variables(self) -> tuple:
        """Compute the variables that depend on the state variables"""

        # Unpack state variables
        L, A, G, T, P, K, S = self.state

        # Recurrent variables for simpler computation
        Xb = self.params['aB'] * L ** 2.
        Xf = self.params['aF'] * G ** 2.
        Xr = self.params['aR'] * S ** 2.
        X = Xb + Xf + Xr
        Z = (P * K) ** (2 / 5) / X ** (4 / 5)

        # derived variables
        B = Xb * Z / self.params['eB']
        F = Xf * Z / self.params['eF']
        R = Xr * Z
        Y = self.params['yE'] * (self.params['eB'] * B + self.params['eF'] * F + R)
        W = (1. - self.params['i']) * Y / P + self.params['wL'] * L / self.params['Sigma']

        return B, F, R, Y, W

    def adjust_parameters(self, action_idx=0):
        """Adjust the parameters before computing the ODE by using the actions"""
        aR_default = 7e-18
        aF_default = 5e6
        aB_default = 3e5
        action = self.action_space[int(action_idx)]
        # subsidy
        if action[0]:
            self.params['aR'] = aR_default * (1 + self.ren_sub)
        else:
            self.params['aR'] = aR_default
        # carbon tax
        if action[1]:
            self.params['aB'] = aB_default * (1 - self.carbon_tax)
            self.params['aF'] = aF_default * (1 - self.carbon_tax)
        else:
            self.params['aB'] = aB_default
            self.params['aF'] = aF_default
            # nature protection
        if action[2]:
            self.Lprot = True
        else:
            self.Lprot = False

    def normalise(self, cur_state):
        return cur_state/np.array(list(self.initials.values()))

    def get_env(self, env_type):
        pass

    def get_reward_function(self, reward_type):
        """Choosing a reward function"""

        def planet_boundaries():
            _, A, _, _, P, _, _ = self.normalise(self.state)
            _, _, _, _, W = self.compute_derived_variables()
            state_vec = np.array([A, W/18588, P])
            r_t = np.linalg.norm(state_vec - self.PB_vec)
            if self.inside_planetary_boundaries():
                return r_t
            return 0

        return planet_boundaries

    def state_is_final(self) -> bool:
        """Check to see if we are in a terminal state"""

        # We use the short-circuiting functionality of Python for a compact if statement:
        # we end the episode early if we move outside planetary boundaries but not too early
        if (not self.inside_planetary_boundaries() or self.t > 600):
            return True
        return False

    def inside_planetary_boundaries(self) -> bool:
        """Check if we are inside the planetary boundaries"""
        self.adjust_parameters()
        _, A, _, _, P, _, _ = self.normalise(self.state)
        _, _, _, _, W = self.compute_derived_variables()

        if A < self.A_PB and W/18588 > self.W_PB and P > self.P_PB:
            return True
        return False