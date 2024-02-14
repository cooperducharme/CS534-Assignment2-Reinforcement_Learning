import datetime
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('classic')

def read_gridworld(gridworld_path : os.path):
    
    with open(gridworld_path, "r") as txt_file:
        temp_file = txt_file.read().split('\n')
        grid_world = []
        for each_row in temp_file:
            temp_row = each_row.split('\t')
            cur_row = []
            for each_val in temp_row:
                if represents_int(each_val):
                    cur_row.append(int(each_val))
                else:
                    cur_row.append(each_val)
            if len(cur_row) > 1:
                grid_world.append(cur_row)
    return grid_world


def represents_int(s):
    try: 
        int(s)
    except ValueError:
        return False
    else:
        return True
    
def save_list_of_reward(list_of_reward : 'list[int]'):
    list_of_avg_reward = []
    for i in range(len(list_of_reward)):
        list_of_avg_reward.append(round(sum(list_of_reward[:i+1])/(i+1),2))
    data_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute().joinpath('data')
    temp_dict = {
        'reward' : list_of_reward,
        'avg' : list_of_avg_reward
    }
    ts = datetime.datetime.now().strftime("%m_%d_at_%H-%M-%S")
    # print(ts)
    with open(data_path.joinpath(ts+'.json'), 'w') as fp:
        json.dump(temp_dict, fp, indent=2)
    
    return 0

def print_list_of_rewards():
    data_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute().joinpath('data')
    files = os.listdir(data_path)
    files.sort()
    color_lst = ['r','g','b','purple']
    index = 0
    for each_file in files:
        f = open (data_path.joinpath(each_file), "r")
        # Reading from file
        data = json.loads(f.read())
        f.close()
        if not data['include']: continue
        
        # plt.plot(data['reward'], label=data["legend"] + "-reward")
        plt.plot(data['avg'], label=data["legend"], c=color_lst[index])
        index += 1
    plt.legend(loc='lower right')
    plt.ylabel('reward points')
    plt.xlabel('count')
    plt.show()        
    return 0

if __name__ == "__main__":    
    print_list_of_rewards()