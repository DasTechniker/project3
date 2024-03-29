# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement three classic algorithm for 
solving Markov Decision Processes either offline or online. 
These algorithms include: value_iteration, policy_iteration and q_learning.
You will test your implementation on three grid world environments. 
You will also have the opportunity to use Q-learning to control a simulated robot 
in crawler.py

The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached the environment should be (re)initialized by
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
You can also only update the visualization of the v values by
    logger.log(i, v)
"""
#++++++++++++++Assignment completed by Timothy Morse, Daniel Herold, Section 004++++++++++++++

# use random library if needed
import random


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you may want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the max over (or sum of) L1 or L2 norms between the values before and
        after an iteration is small enough. For the Grid World environment, 1e-4
        is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
   
### Please finish the code below ##############################################
###############################################################################
    #Set up storage for v-values one iteration bacck
    v_old = [0]*NUM_STATES

    #For max_iterations...
    for k in range(1,max_iterations+1):
        #Update logger
        logger.log(k, v, pi)

        #For each state, determine best actions
        for s in range(0, NUM_STATES):
            #Set up storage for each action value
            tempVals = [0]*NUM_ACTIONS
            for a in range(0,NUM_ACTIONS): #Iterate through all actions and determine values
                t=env.trans_model[s][a]
                for epitaph in t:
                    tempVals[a] += epitaph[0] * (epitaph[2] + (gamma * v_old[epitaph[1]]))
            v[s]=max(tempVals) #Set v[s] to max action value
            pi[s]=tempVals.index(max(tempVals)) #Set pi[s] to action with highest action value
        v_old=v #Update v_old to v for next iteration

    logger.log(k, v, pi) #Update logger one last time

###############################################################################
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Implement policy iteration to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the max over (or sum of) 
        L1 or L2 norm between the values before and after an iteration is small enough. 
        For the Grid World environment, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of value by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
    #Set up storage for v and pi values one iteration back
    v_old = v
    pi_old = pi

    #Also store whether the policy has converged, current iteration, and a counter for how many iterations the policy hasn't changed in
    converged = False
    k = 1
    unchanged = 0

    #While both not at max iterations and not converged, iterate.
    while not converged and k!=max_iterations+1:
        logger.log(k, v, pi) #Update logger, k
        k+=1

        #For each state, get a value for the current policy's action
        for s in range(0, NUM_STATES):
            tempVals = [0] * NUM_ACTIONS
            t = env.trans_model[s][pi[s]]
            for epitaph in t:
                tempVals[pi[s]] += epitaph[0] * (epitaph[2] + (gamma * v_old[epitaph[1]]))
            v[s] = max(tempVals)

        #After values are gotten, do limited value iteration to improve the policy
        for s in range(0, NUM_STATES):
            tempVals = [0] * NUM_ACTIONS
            for a in range(0, NUM_ACTIONS):
                t = env.trans_model[s][a]
                for epitaph in t:
                    tempVals[a] += epitaph[0] * (epitaph[2] + (gamma * v_old[epitaph[1]]))
            if v[s] < max(tempVals):
                v[s] = max(tempVals)
                pi[s] = tempVals.index(max(tempVals))

        #If pi hasn't changed, update unchanged counter, else, reset counter
        if pi_old == pi:
            unchanged+=1
        else:
            unchanged=0
        #If uunchanged counter is at 15 or above, stop iteration
        if unchanged>=15:
            converged=True

        #Update v_old and pi_old
        v_old = v
        pi_old = pi

###############################################################################
    return pi


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (training episodes) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = .8
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

    

### Please finish the code below ##############################################
###############################################################################
    #Initialise q and q_old
    q = [0]*NUM_STATES
    for i in range(NUM_STATES):
        q[i] = [0]*NUM_ACTIONS
    q_old = q

    for k in range(1,max_iterations+1):
        logger.log(k, v, pi) #Update logger
        s = env.reset() #Initialise s
        stillGo = True #Set up control boolean - there's other ways, but this works.
        while stillGo: #While not at a terminal state, keep going
            a = 0 #If all action values same OR at a random chance, pick random action. Else, pick max action.
            if (sum(q[s])/len(q[s]))==max(q[s]) or random.random()>=eps:
                a = random.randrange(NUM_ACTIONS)
            else:
                a = q[s].index(max(q[s]))
            s_, r, terminal, info = env.step(a) #Get results of actions
            stillGo = not terminal #If terminal state,update control boolean
            q[s][a]=(q_old[s][a])+(alpha*(r+(gamma*max(q_old[s_]))-q_old[s][a])) #Update q[s][a]
            v[s]=max(q[s]) #Update v[s] with max value of q[s]
            pi[s]=q[s].index(max(q[s])) #Update pi[s] with max action of q[s]
            s = s_ #Update s to s_
            q_old = q #Update q_old
    logger.log(k, v, pi) #Update logger one last time
###############################################################################
    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            [10, "s", "s", "s", 1],
            [-10, -10, -10, -10, -10],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()