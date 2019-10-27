# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this file, you should test your Q-learning implementation on the robot crawler environment 
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""


# use random library if needed
import random

def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = .8
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 2500
    #########################

### Please finish the code below ##############################################
###############################################################################
    v = [0]*NUM_STATES
    pi = [0]*NUM_STATES
    q = [0] * NUM_STATES
    for i in range(NUM_STATES):
        q[i] = [0] * NUM_ACTIONS
    q_old = q
    print(str(q))
    s = env.reset()
    for k in range(1, max_iterations):
        logger.log(k, v, pi)

        stillGo = True
        visited = [False] * NUM_STATES
        mustRand = False
        visited[s] = True
        a = 0
        if (sum(q[s]) / len(q[s])) == max(q[s]) or random.random() >= eps or mustRand:
            a = random.randrange(NUM_ACTIONS)
        else:
            a = q[s].index(max(q[s]))
        s_, r, terminal, info = env.step(a)
        stillGo = not terminal
        q[s][a] = (q_old[s][a]) + (alpha * (r + (gamma * max(q_old[s_])) - q_old[s][a]))
        v[s] = max(q[s])
        print(s)
        pi[s] = q[s].index(max(q[s]))
        s = s_
        q_old = q
###############################################################################
    return pi


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()