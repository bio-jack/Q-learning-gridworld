import numpy as np
import matplotlib.pyplot as plt

def generateQTable():
  """
  Initialises an empty 4x12 table of Q-Values where rows are states and columns 
  are actions.

  Parameters - None
  
  Returns - q_table (np.array)
  
  """
  q_table = np.zeros((48, 4))
  return q_table


def eGreedyAction(q_table, state, e=0.1):
  
  """
  Returns max Q-Value for a given state with probability 1 - e. Returns a random
  exploratory action with probability e.

  Parameters
  ----------
  q_table (np.array)
    Input table of Q-Values
  state (int)
    Value associated with particular state
  e (float)
    Epsilon value determining probability of exploratory action
  
  Returns
  ---------- 
  action (int)
    0 = Up, 1 = Right, 2 = Down, 3 = Left

  """
  # Greedy action
  if np.random.default_rng().random() > e:
    action = np.random.choice(np.where(q_table[state, :] == q_table[state, :].max())[0])
    return action
  
  # Exploratory action
  else:
    action = np.random.default_rng().integers(0, 4, size=1)[0]
    return action


def getState(agent):
  """
  Calculates state associated with (x, y) co-ordinates of an agent.

  Parameters
  ----------
  agent (tuple)
    Tuple of (x, y) co-ordinates
  
  Returns
  ----------
  state (int)
    State associated with agent co-ordinates

  """
  x, y = agent
  state = 12 * x + y
  return state


def moveAgent(agent, action):
  """
  Changes agent (x, y) co-ordinates dependent on given action. Does not change
  co-ordinates if action would move agent off the board.

  Parameters
  ----------
  agent (tuple)
    Tuple of (x, y) co-ordinates
  action (int)
    0 = Up, 1 = Right, 2 = Down, 3 = Left
  
  Returns
  ----------
  agent (tuple)
    Tuple of (x, y) co-ordinates
  
  """
  x, y = agent

  if action == 0 and x > 0:     # Up
    x -= 1
  elif action == 1 and y < 11:  # Right
    y += 1
  elif action == 2 and x < 3:   # Down
    x += 1
  elif action == 3 and y > 0:   # Left
    y -= 1
  
  agent = x, y

  return agent


def getReward(state):
  """
  Returns reward from reaching a given state. States 37 - 46 are "cliff" states
  with -100 reward and state 47 is the terminal state associated with reward 0
  as well as ending the episode. All other states have reward -1.

  Parameters
  ----------
  state (int)
    Value associated with particular state
  
  Returns
  ----------
  reward (tuple)
    Tuple where index 0 is the reward and index 1 is the game state - i.e.
    whether the game is over

  """
  game_over = False
  # Cliff states
  if state >= 37 and state <= 46:
    reward = -100
    game_over = False
    return reward, game_over
  # Terminal state
  elif state == 47:
    reward = 0
    game_over = True
    return reward, game_over
  # Other states
  else:
    reward = -1
    return reward, game_over


def getQValueForState(state, q_table, action):
  """
  Returns Q-Value for a given state from Q-Table.
  
  Parameters - state (int), q_table (np.array), action (int)
  Returns - float
  """
  return q_table[state, action]


def getMaxQValueForState(state, q_table):
  """
  Returns maximum Q-Value for a given state from Q-Table.
  
  Parameters - state(int), q_table(np.array)
  Returns - float
  """
  max_index = np.random.choice(np.where(q_table[state, :] == q_table[state, :].max())[0])
  return q_table[state, max_index]


def updateQTable(q_table, oldState, nxtValue, reward, action):
  """
  Implements formula to update Q-Values depending on algorithm (Q-Learning or 
  SARSA).

  Parameters
  ----------
  q_table (np.array)
    Input table of Q-Values
  oldState (int)
    Value associated with state first action is taken from
  nxtValue (int)
    Q-Value associated with either max action taken from next state (Q-Learning)
    or action dictated by policy in the next state (SARSA).
  reward (int)
    Reward associated with moving to new state.
  action (int)
    0 = Up, 1 = Right, 2 = Down, 3 = Left

  Returns
  ----------
  updated_q_table (np.array)
    Output table of Q-Values with update.

  """
  updated_q_table = q_table
  new_q_value = q_table[oldState, action] + 0.5 * (reward[0] + 0.9 * nxtValue - q_table[oldState, action])
  updated_q_table[oldState, action] = new_q_value

  return updated_q_table


def qLearning():
  """
  Implements Q-Learning algorithm for cliff-walking task with parameters from 
  Sutton and Barto.

  Parameters
  ---------- 
  None
  
  Returns
  ----------
  cumulative_rewards (list)
    List of rewards obtained for each episode.

  """
  cumulative_rewards = []

  # Grid for tracing agent path
  grid = np.zeros((4, 12))
  
  q_table = generateQTable()
  
  # Iterate over 500 episodes
  for episode in range(500):
    # Cumulative reward for single episode
    cumulative_reward = 0
    # Initialise agent (x, y) co-ordinates
    agent = (3, 0)
    
    # Loop until game is over
    while True:
      oldState = getState(agent)
      action = eGreedyAction(q_table, oldState)
      agent = moveAgent(agent, action)
      
      # Trace agent path on last episode
      if episode == 499:
        x, y = agent
        grid[x, y] = 1

      newState = getState(agent)
      reward = getReward(newState)
      cumulative_reward += reward[0]
      # Get action with maximum Q-Value for new state
      maxNxtQValue = getMaxQValueForState(newState, q_table)
      q_table = updateQTable(q_table, oldState, maxNxtQValue, reward, action)
      
      # End game if terminal state reached
      if reward[1] == True: # if state = 47
        break
      # Reset agent to state 36 if cliff encountered, do not end episode
      elif newState <= 46 and newState >= 37:
        agent = (3, 0)

    # Append reward for current episode to list of rewards for all episodes
    cumulative_rewards.append(cumulative_reward)

  print("Q-Learning agent path on episode 500:")
  print(grid)

  return cumulative_rewards


def SARSA():
  """
  Implements SARSA algorithm for cliff-walking task with parameters from 
  Sutton and Barto.

  Parameters
  ----------
  None

  Returns
  ----------
  cumulative_rewards (list)
    List of rewards obtained for each episode.

  """
  cumulative_rewards = []

  # Grid for tracing agent path
  grid = np.zeros((4, 12))
  
  q_table = generateQTable()
  
  # Iterate over 500 episodes
  for episode in range(500):
    # Cumulative reward for single episode
    cumulative_reward = 0
    # Initialise agent (x, y) co-ordinates
    agent = (3, 0)

    state = getState(agent)
    action = eGreedyAction(q_table, state)

    while True:

      oldState = getState(agent)
      agent = moveAgent(agent, action)
      newState = getState(agent)

      if episode == 499:
        x, y = agent
        grid[x, y] = 1

      reward = getReward(newState)
      cumulative_reward += reward[0]

      # Get action for new state dictated by e-greedy policy
      new_state_action = eGreedyAction(q_table, newState)
      # Get Q-Value associated with action
      new_state_q_value = getQValueForState(newState, q_table, new_state_action)

      q_table = updateQTable(q_table, oldState, new_state_q_value, reward, action)
      
      # Ensure next action is action for which we obtained Q-Value previously
      action = new_state_action
      
      # End game if terminal state reached
      if reward[1] == True: # if state = 47
        break
      # Reset agent to state 36 if cliff encountered, do not end episode
      elif newState <= 46 and newState >= 37:
        agent = (3, 0)
    # Append reward for episode to list of rewards for all episdoes
    cumulative_rewards.append(cumulative_reward)

  print("SARSA agent path on episode 500:")
  print(grid)

  return cumulative_rewards


def smoothOutput(cumulative_rewards):
  """
  Smooths cumulative reward output data by averaging per 10 episodes over the
  500 episodes.

  Parameters
  ----------
  cumulative_rewards (list)
    List of cumulative rewards for each episode
  
  Returns
  ----------
  smoothed_output (list)
    List of cumulative rewards averaged over sequential batches of 10 episodes

  """
  smoothed_output = []
  
  posA = 0
  posB = 9
  
  for _ in range(50):
    smoothed_output.append(sum(cumulative_rewards[posA: posB])/10)
    posA += 10
    posB += 10

  return smoothed_output


def plotCumRewards(q_learning_cum_rewards=None, sarsa_cum_rewards=None):
  """
  Plots cumulative rewards per episode for Q-Learning and SARSA algorithms on 
  the cliff walking task.

  Parameters
  ----------
  q_learning_cum_rewards (list)
    Output of Q-Learing algorithm

  sarsa_cum_rewards (list)
    Output of SARSA algorithm

  """
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.set_xlabel("Episode")
  ax.set_ylabel("Cum reward per episode")
  ax.set_ylim([-250, 0])
  ax.plot(q_learning_cum_rewards, label = "Q-Learning")
  ax.plot(sarsa_cum_rewards, label = "SARSA")
  fig.legend()
  plt.show()

if __name__ == "__main__":
  # Run Q-Learning and SARSA algorithms
  qLearnOutput = qLearning()
  sarsaOutput = SARSA()
  # Smooth cumulative reward output and plot
  smoothedQ = smoothOutput(qLearnOutput)
  smoothedSARSA = smoothOutput(sarsaOutput)
  plotCumRewards(smoothedQ, smoothedSARSA)
  