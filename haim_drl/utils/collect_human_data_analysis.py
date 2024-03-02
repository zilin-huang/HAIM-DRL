import json
import numpy as np
import pandas as pd

# Load data
with open('/home/zilin/code/HAIM-DRL/haim_drl/utils/human_traj_100_0.06.json', "r") as f:
    source = json.load(f)
    source_keys = source.keys()
    # Printing keys in the source JSON file
    print("Keys in the JSON file:", source_keys)

# Uncomment the below lines to print the entire source JSON
# print(source)

# episode_reward = source["episode_reward"]
# Uncomment the below lines to print the episode rewards and related stats
# print("Episode rewards:", episode_reward)
# print("Average episode reward:", np.mean(episode_reward))
# print("Number of episode rewards:", len(episode_reward))

# Extracting pool data
pool = source["data"]
# Printing the number of items in the pool
print("Number of items in the pool:", len(pool))

# Uncomment the below line to print the 800th pool item
# print("Pool item 800:", pool[800])

# Count the number of times 'disturbance' is True
disturbance_count = sum([item['infos']['disturbance'] for item in pool])
# Calculate disturbance rate
disturbance_rate = disturbance_count / len(pool)
# Printing disturbance rate
print('Disturbance Rate:', disturbance_rate)
print('disturbance_count', disturbance_count)

# Printing the success rate
success_rate = source["success_rate"]
print("Success rate:", success_rate)

episode_reward = source["episode_reward"]
# print("episode_reward:", episode_reward)
print("Average episode cost:", np.mean(episode_reward))

episode_cost = source["episode_cost"]
print("Episode costs:", episode_cost)
print("Average episode cost:", np.mean(episode_cost))
print("Sum of episode costs:", np.sum(episode_cost))

# episode_disturbance_cost = source["episode_disturbance_cost"]
# Uncomment the below lines to print the episode disturbance cost and related stats
# print("Episode disturbance costs:", episode_disturbance_cost)
# print("Average episode disturbance cost:", np.mean(episode_disturbance_cost))
# print("Sum of episode disturbance costs:", np.sum(episode_disturbance_cost))

# Calculate and print total episode length
episode_len = source["episode_len"]
print("Total episode length:", sum(episode_len))



#
#
#
# first_pool = pool[31708]
# print(first_pool)
#
# first_obs = first_pool["obs"]
# print (first_obs)
# print(len(first_obs))

#
#
# obs= ["data"]["obs"]
# print(len(obs[1]))

#
#
# episode_reward = source["episode_reward"]
# print(len(episode_reward))
# print(episode_reward)
#
# success_rate = source["success_rate"]
# print(success_rate)
#
# episode_len = source["episode_len"]
# print(episode_len)
#
# episode_cost = source["episode_cost"]
# print(episode_cost)
#
#
# print(pool[1])

# for i in range(1):
#     print(pool[i])
#

# # 打印前10个数据样本
# for i in range(10):
#     print(data['data'][i])