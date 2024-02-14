from cmath import sqrt
import copy
from math import gamma, log
import time
import random

from src.helper import save_list_of_reward

percent_time_exploring = 0.3
starting_alpha = 0.1
ending_alpha = 0.05
starting_gamma = 0.9
ending_gamma = 0.2
starting_epsilon = 1
ending_epsilon = 0.05

epsilon_factor = (ending_epsilon/starting_epsilon)**0.001
gamma_factor = (ending_gamma/starting_gamma)**0.001
alpha_factor = (ending_alpha/starting_alpha)**0.001

gama = starting_gamma
alpha = starting_alpha
epsilon = starting_epsilon

class State():
    def __init__(self, row : int, col : int):
        self.row = row
        self.col = col

    # def __eq__(self, other):
    #     return self.row == other.row and self.col == other.col
class Action():
    def __init__(self, d_row : int, d_col : int):
        self.d_row = d_row
        self.d_col = d_col

class Q_table():
    ACTION_TRANSLATOR = {
        (-1,0): 0,
        (1,0): 1,
        (0,-1): 2,
        (0,1): 3
    }
    
    def __init__(self, gridworld):
        q_table = []
        visit_table = []
        row = len(gridworld)
        col = len(gridworld[0])
        # initialze q-table
        for each_row in range(row):
            cur_row = []
            for each_col in range(col):
                cur_ele = [0]*4
                cur_row.append(cur_ele)
            q_table.append(cur_row)
            visit_table.append([0]*col)
        self.q_table = q_table
        self.visit_table = visit_table
        self.visit_table_cache = visit_table
    
    def reset_visited(self):
        self.visit_table_cache = [[sum(pair) for pair in zip(*pairs)] for pairs in zip(self.visit_table_cache, self.visit_table)]
        for each_row in range(len(self.visit_table)):
            for each_col in range(len(self.visit_table[0])):
                self.visit_table[each_row][each_col] = 0

    def update_table(self, state : State, action : Action, q_val : float) -> None:
        row = state.row
        col = state.col
        action_index = self.ACTION_TRANSLATOR[(action.d_row,action.d_col)]
        self.q_table[row][col][action_index] = q_val
        self.visit_table[row][col] += 1
    
    def get_q_val(self, state : State, action : Action) -> float:
        row = state.row
        col = state.col
        action_index = self.ACTION_TRANSLATOR[(action.d_row,action.d_col)]
        return self.q_table[row][col][action_index]
    
    def get_highest_q_val(self, state : State, action : Action) -> float:
        # TODO might not be correct
        row = state.row
        col = state.col
        return max(self.q_table[row][col])
    
    #print_q_table
    def print_q_table(self) -> None:
        # print('Q table with value in up, down, left, right sequence:', *self.q_table, sep='\n- ')
        print('Q table with value in up, down, left, right sequence:')
        print('\n'.join('Row {}: {}'.format(*k) for k in enumerate(self.q_table)))

    #print_policy        
    def print_policy(self) -> None:
        policy = []
        action_translate_dict = {
            0: '^',
            1: 'V',
            2: '<',
            3: '>'
        }
        row = len(self.q_table)
        col = len(self.q_table[0])
        
        global cur_gridworld
        for each_row in range(row):
            cur_row = []
            for each_col in range(col):
                cur_ele = self.q_table[each_row][each_col]
                max_ele_index = cur_ele.index(max(cur_ele))
                if max(cur_ele) > 0.0:
                    action_str = action_translate_dict[max_ele_index]
                    cur_gw_ele = cur_gridworld[each_row][each_col]
                    if cur_gw_ele == 'S' or cur_gw_ele == '+' or cur_gw_ele == '-' or (isinstance(cur_gw_ele,str) and cur_gw_ele.islower()):
                        cur_row.append(action_str + cur_gw_ele + action_str)
                    else:
                        cur_row.append(action_str)
                else:
                    cur_row.append(cur_gridworld[each_row][each_col])
            policy.append(cur_row)
        # print('Policy table:', *policy, sep='\n')
        print('Policy table:')
        s = [[str(e) for e in row] for row in policy]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print ('\n'.join(table))
        
    def print_heatmap(self) -> None:
        heatmap = []
        row = len(self.visit_table_cache)
        col = len(self.visit_table_cache[0])
        total_visits = 0
        for each_row in range(row):
            for each_col in range(col):
                total_visits+=self.visit_table_cache[each_row][each_col]
        for each_row in range(row):
            cur_row = []
            for each_col in range(col):
                cur_row.append(self.visit_table_cache[each_row][each_col]/total_visits)
            heatmap.append(cur_row)
        print("Heat Map:")
        s = [[str(round(e*100)) for e in row] for row in heatmap]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print ('\n'.join(table))

cur_gridworld = [[]]
q_table = Q_table(cur_gridworld)
start_state = None
highest_terminal = None

def valid_state(row : int,col : int) -> bool:
    global cur_gridworld
    global q_table
    if col < 0 or row < 0 or row >= len(cur_gridworld) or col >= len(cur_gridworld[0]):
        return False
    if cur_gridworld[row][col] == 'X' or cur_gridworld[row][col] == 'A':
        return False
    else:
        return True

def available_actions(state : State) -> list:
    row = state.row
    col = state.col
    list_of_actions = []
    for (d_row,d_col) in (-1,0),(1,0),(0,-1),(0,1):
        cur_row = row + d_row
        cur_col = col + d_col
        if valid_state(cur_row,cur_col):
            list_of_actions.append(Action(d_row,d_col))
    return list_of_actions
        
def random_action(state : State) -> Action:
    list_of_actions = available_actions(state)
    if len(list_of_actions) == 0:
        return None
    action_index = random.randrange(len(list_of_actions))
    rand_action = list_of_actions[action_index]
    return rand_action

def highest_Q_action(state : State) -> Action:
    cur_highest_q = -1
    cur_action = None
    list_of_actions = available_actions(state)
    global q_table
    for each_action in list_of_actions:
        cur_q_val = q_table.get_q_val(state,each_action)
        if cur_q_val > cur_highest_q:
            cur_highest_q = cur_q_val
            cur_action = each_action
    return cur_action

def highest_UCB(state: State, iterations : int) ->Action:
    cur_highest_UCB = -1
    list_of_actions = available_actions(state)
    global q_table
    for each_action in list_of_actions:
        if q_table.visit_table[state.row+each_action.d_row][state.col+each_action.d_col] == 0:
            return each_action
        else:
            cur_UCB = q_table.get_q_val(state,each_action) + sqrt(log(iterations + 1)/q_table.visit_table[state.row+each_action.d_row][state.col+each_action.d_col]).real
            if cur_UCB > cur_highest_UCB:
                cur_highest_UCB = cur_UCB
                cur_action = each_action
    return cur_action

def selectAction(state : State, iteration : int) -> Action:
    """/* will want to make exploration more complex */
        if rand() < epsilon
            return random action
        else
            return action with highest Q(s,a) value
    """
    global cur_gridworld
    global epsilon

    if random.random() < epsilon: # (0,1)
        return highest_UCB(state,iteration)
        #return random_action(state)
    else:
        return highest_Q_action(state)

def takeAction(state : State , action : Action, success : int) -> State: # trickier :-) 
    """/* the ONLY PLACE the transition model should appear in your code 
    For Part 1 can just perform the action correctly */
    """
    prob = random.random()
    move_succeed =  (prob <= success)
    
    if move_succeed:
        s_prime = State(state.row + action.d_row, state.col + action.d_col)
    else:
        dir_fail = random.random()
        forward = (dir_fail <= .5)
        if forward:
            s_intermediate = State(state.row + action.d_row, state.col + action.d_col)
            if not valid_state(s_intermediate.row, s_intermediate.col):
                return state
            s_prime = State(state.row + (2*action.d_row), state.col + (2*action.d_col))
        else:
            s_prime = State(state.row - action.d_row, state.col - action.d_col)
    if valid_state(s_prime.row,s_prime.col):
        return s_prime
    else:
        return state

def remove_wall(letter : str):
    global cur_gridworld
    upper_letter = letter.upper()
    row = len(cur_gridworld)
    col = len(cur_gridworld[0])
    for each_row in range(row):
        for each_col in range(col):
            cur_val = cur_gridworld[each_row][each_col]
            if cur_val == letter or cur_val == upper_letter:
                cur_gridworld[each_row][each_col] = 0

def get_reward(state : State):
    global cur_gridworld
    val = cur_gridworld[state.row][state.col]
    if isinstance(val,int):
        return val
    elif val == 'S':
        return 0
    elif val == '+':
        cur_gridworld[state.row][state.col] = 0
        return 2
    elif val == '-':
        cur_gridworld[state.row][state.col] = 0
        return -2
    elif val == 'a':
        remove_wall(val)
        return 0
    elif val == 'X' or val == 'A':
        print('No the agent should not wander into the wall!')

def update_SARSA(state : State, action : Action , s_prime : State, a_prime : Action, iteration : int) -> float: # /* depends on SARSA vs Q-learning */
    global alpha
    global gama
    global q_table

    q_s_a = q_table.get_q_val(state,action)
    q_s_prime_a_prime = q_table.get_q_val(s_prime,a_prime)
    s_prime_reward = get_reward(s_prime)
    q_s_a_new = round(q_s_a + alpha*(s_prime_reward + gama*q_s_prime_a_prime - q_s_a),2)
    q_table.update_table(state,action,q_val=q_s_a_new)
    return s_prime_reward

def update_Q_learning(state : State,action : Action ,s_prime : State,iteration : int) -> float: # /* depends on SARSA vs Q-learning */
    global alpha
    global gama
    global q_table

    q_s_a = q_table.get_q_val(state,action)
    q_s_prime_a = q_table.get_highest_q_val(s_prime,action)
    s_prime_reward = get_reward(s_prime)
    q_s_a_new = round(q_s_a + alpha*(s_prime_reward + gama*q_s_prime_a - q_s_a),2)
    q_table.update_table(state,action,q_val=q_s_a_new)
    return s_prime_reward

def notTerminal(state : State):
    global cur_gridworld
    global highest_terminal
    val = cur_gridworld[state.row][state.col]
    if not isinstance(val,int) or val == 0:
        return True
    else:
        if highest_terminal == None:
            highest_terminal = val
        else:
            if val > highest_terminal:
                highest_terminal = val
        return False

def startState():
    global start_state
    # start_state = State(6,1)
    # return start_state
    if start_state != None:
        return start_state
    else:
        global cur_gridworld
        for row in range(len(cur_gridworld)):
            for col in range(len(cur_gridworld[0])):
                val = cur_gridworld[row][col]
                if val == "S":
                    start_state = State(row,col)
                    return start_state

def percent_visited():
    global q_table
    total_visited = 0
    for each_row in range(len(q_table.visit_table_cache)):
        for each_col in range(len(q_table.visit_table_cache[0])):
            if q_table.visit_table_cache[each_row][each_col] != 0:
                total_visited += 1
    total_size = len(q_table.visit_table_cache) * len(q_table.visit_table_cache[0])
    return total_visited/total_size

def rl(gridworld: list, action_reward : float, prob : float, time_limit : float, SARSA_flag : bool) -> None:
    if SARSA_flag:
        q_table = SARSA(gridworld, action_reward, prob, time_limit)
    else:
        q_table = q_learning(gridworld, action_reward, prob, time_limit)
        
    # output policy table at the end
    global cur_gridworld
    cur_gridworld = copy.deepcopy(gridworld)
    q_table.print_policy()
    q_table.print_heatmap()

def q_learning(gridworld: list, action_reward : float, prob : float, time_limit : float) -> Q_table:
    global cur_gridworld
    global q_table
    global start_state
    global highest_terminal
    global epsilon
    global alpha
    global gama
    global alpha_factor
    global gamma_factor
    global epsilon_factor
    global percent_time_exploring

    exploit = False
    start_time = time.time()
    cur_gridworld = copy.deepcopy(gridworld)
    q_table = Q_table(cur_gridworld)
    iteration = 0
    list_of_reward = []
    cur_time = time.time()
    update_time = (time_limit*(1-percent_time_exploring))/1000
    update_timer = time.time()
    while time.time()-start_time < time_limit:
        # Reset the state (New search)
        s = startState()
        iteration += 1
        cur_reward = 0
        # Ensure cookies and glass are restored
        cur_gridworld = copy.deepcopy(gridworld)
        q_table.reset_visited()
        while notTerminal(s): # terminal state
            # Determine the next action
            a = selectAction(s, iteration)
            # print(a)
            if a == None:
                print("No action selected")
                break
            # Take the next action
            s_prime = takeAction(s, a, prob)
            # Update the Q-value for the current state-action pair
            s_prime_reward = update_Q_learning(s, a, s_prime, iteration)  
            cur_reward += s_prime_reward + action_reward
            if s == s_prime : break
            # Update the state
            s = s_prime
        if (time.time() - update_timer) > update_time:
            if exploit:
                epsilon*= epsilon_factor
                alpha*= alpha_factor
                gama*= gamma_factor
                update_timer = time.time()
            if (time.time() - cur_time) > 0.1:
                if percent_visited() > 0.90:
                    exploit = True
                elif highest_terminal > 7:
                    exploit = True
                elif (time.time()-start_time) > time_limit*0.3 and highest_terminal:
                    exploit = True
                cur_time = time.time()
                list_of_reward.append(cur_reward)
                print("average reward for iteration # {} is : {}".format(iteration, round(sum(list_of_reward)/len(list_of_reward),2)))
                # print("cur reward: ", cur_reward)
    # save_list_of_reward(list_of_reward)
    return q_table

def SARSA(gridworld: list, action_reward : float, prob : float, time_limit : float) -> Q_table:
    global cur_gridworld
    global q_table
    global start_state
    global highest_terminal
    global epsilon
    global alpha
    global gama
    global alpha_factor
    global gamma_factor
    global epsilon_factor
    global percent_time_exploring

    exploit = False
    start_time = time.time()
    cur_gridworld = copy.deepcopy(gridworld)
    q_table = Q_table(cur_gridworld)
    iteration = 0
    list_of_reward = []
    cur_time = time.time()
    update_time = (time_limit*(1-percent_time_exploring))/1000
    update_timer = time.time()
    while time.time()-start_time < time_limit:
        # Reset the state (New search)
        iteration += 1
        cur_reward = 0
        s = startState()
        a = selectAction(s, iteration)
        # Ensure cookies and glass are restored
        cur_gridworld = copy.deepcopy(gridworld)
        q_table.reset_visited()
        while notTerminal(s): # terminal state
            if a == None:
                break
            # Take the next action
            s_prime = takeAction(s, a, prob)
            # Determine the next action
            a_prime = selectAction(s_prime, iteration)
            if a_prime == None:
                break
            # Update the Q-value for the current state-action pair
            s_prime_reward = update_SARSA(s, a, s_prime, a_prime, iteration) 
            cur_reward += s_prime_reward + action_reward
            if s == s_prime : break
            # Update the state
            s = s_prime
            a = a_prime
        if (time.time() - update_timer) > update_time:
            if exploit:
                epsilon*= epsilon_factor
                alpha*= alpha_factor
                gama*= gamma_factor
                update_timer = time.time()
            if (time.time() - cur_time) > 0.1:
                if percent_visited() > 0.90:
                    exploit = True
                elif highest_terminal > 7:
                    exploit = True
                elif (time.time()-start_time) > time_limit*0.3 and highest_terminal:
                    exploit = True
                cur_time = time.time()
                list_of_reward.append(cur_reward)
                print("average reward for iteration # {} is : {}".format(iteration, round(sum(list_of_reward)/len(list_of_reward),2)))
                # print("cur reward: ", cur_reward)
    # save_list_of_reward(list_of_reward)
    return q_table
    
