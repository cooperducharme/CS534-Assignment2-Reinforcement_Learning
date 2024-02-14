import os
import sys

from src.rl import rl
from src.epsilon_greedy import epsilon_greedy
from src.helper import read_gridworld

def main(gridworld_name : str, action_reward : float, p_correct : float, time_limit : float, epsilon_flag : bool, SARSA_flag : bool):
    gridworld_dir = os.path.dirname(os.path.realpath(__file__)) + '/gridworlds/'
    gw = read_gridworld(gridworld_dir + gridworld_name)
    if epsilon_flag:
        epsilon_greedy(gw, action_reward, p_correct, time_limit, SARSA_flag)
    else:
        rl(gw, action_reward, p_correct, time_limit, SARSA_flag)
    return 0

def read_args(argv):

    command_format = "\ncommands: [filename] [seconds_to_run] [action_reward] [p_action_success] [time_based?]"

    print(command_format)

    while True:
        argv = get_input()

        if len(argv) != 0 and argv[0] == "q":
            return

        if len(argv) < 5:
            print("Not enough command line arguments", command_format)
            continue
        
        try:
            gridworld_name = argv[0]
            time_limit = float(argv[1])
            action_reward = float(argv[2])
            p_correct = float(argv[3])
            epsilon_flag = argv[4]
            SARSA_flag = False
            epsilon_flag = False if (epsilon_flag.lower() == "true" or epsilon_flag.lower() == "t") else True
        except ValueError:
            print("Invalid input, please try again.")
            continue
        
        args = {
            "gridworld_name": gridworld_name,
            "action_reward": action_reward,
            "p_correct": p_correct,
            "time_limit": time_limit,
            "epsilon_flag": epsilon_flag,
            "SARSA_flag": SARSA_flag
        }
        
        print(args)
        main(**args)


def get_input():
    user_input = input("Enter arguments (enter q to quit):")
    return user_input.split()


if __name__ == "__main__":
    read_args(sys.argv[1:])