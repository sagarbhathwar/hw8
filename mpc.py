import os
import numpy as np
import gym
import sapien_rl.env
import copy
import argparse

class MPC:
    def __init__(self, env, plan_horizon = 4, popsize = 100, num_elites = 20, max_iters = 4, use_mpc = True):
        """
        :param env:
        :param plan_horizon: 
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param use_mpc: Whether to use only the first action of a planned trajectory
        """
        self.env = env
        self.use_mpc = use_mpc
        self.plan_horizon = plan_horizon
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites
        self.action_dim = env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low # used to clip your actions
        self.reset()

    def reset(self):
        self.mean = np.zeros((self.plan_horizon * self.action_dim))
        self.std = 0.5 * np.ones((self.plan_horizon * self.action_dim))
        
    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        next_states = []
        rewards = []
        for i in range(len(states)):
            next_state, reward, info = self.env.mpc_step(states[i], actions[i])
            next_states.append(next_state)
            rewards.append(reward)
        return next_states, rewards

    def cem_optimize(self, state):
        mean = self.mean.copy()
        std = self.std.copy()
        initial_state = state.copy()

        for i in range(self.max_iters):

            next_states = np.repeat(initial_state[np.newaxis], self.popsize, axis=0)

            actions = np.random.normal(mean, std,
                size=(self.popsize, self.plan_horizon * self.action_dim))

            rewards = np.empty((self.popsize, self.plan_horizon))

            for t in range(self.plan_horizon):
                action_dim = self.action_dim
                actions_to_take = actions[:, t * action_dim:(t + 1) * action_dim]
                next_states, next_rewards = self.predict_next_state_gt(next_states, actions_to_take)
                rewards[:, t] = next_rewards

            sor = rewards.sum(axis=1)
            elites = actions[np.argsort(sor)][-self.num_elites:]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0)


        return mean, std

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        if self.use_mpc == False:
            if t % self.plan_horizon == 0:
                self.mean, self.std = self.cem_optimize(state)
            idx = (t % self.plan_horizon) * self.action_dim
            action = self.mean[idx:idx + self.action_dim]
        else:
            self.mean, self.std = self.cem_optimize(state)
            action = self.mean[:self.action_dim]  # First action

            self.mean[:-self.action_dim] = self.mean[self.action_dim:]
            self.mean[-self.action_dim:] = 0
            self.std[:-self.action_dim] = self.std[self.action_dim:]
            self.std[-self.action_dim:] = 0.5

        return action
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0, help='save index')
    opt = parser.parse_args()
    
    env = gym.make('OpenCabinet_state_45267_link_0-v4')



    env.reset(level = 3)
    env.perturb_gripper()
    obs = env.get_obs()
    
    # cem_mpc = MPC(env, use_mpc=False, max_iters=1, popsize = 400, num_elites=1)
    cem_mpc = MPC(env, use_mpc=False)
    states = []
    actions = []
    next_states = []
    
    # env.render('human')
    for t in range(150):
        state = env.get_state("world")
        action = cem_mpc.act(state, t)
        states.append(state)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        next_states.append(obs)
        print("---step #%d----" % t)
        print(info['state_info'])
        print("reward: %f" % reward, info['eval_info'])
        if info['eval_info']['success']:
            print("Bravo! Open the door!")
            break
        if done:
            break
    
    # save the transitions
    np.save("trajectories/%d_traj.npy" % opt.id, {"states": states, "actions": actions, "next_states": next_states})