import os.path

from haim_drl.algo.haim_drl.haim_drl import HAIM_DRLTrainer
from haim_drl.utils.human_in_the_loop_env import HumanInTheLoopEnv
from haim_drl.utils.train_utils import initialize_ray

import csv


def get_function(exp_path, ckpt_idx):
    ckpt = os.path.join(exp_path, "checkpoint_{}".format(ckpt_idx), "checkpoint-{}".format(ckpt_idx))
    trainer = HAIM_DRLTrainer(dict(env=HumanInTheLoopEnv))

    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs})
        return ret

    return _f


if __name__ == '__main__':
    # hyperparameters
    CKPT_PATH ='/home/zilin/code/HAIM-DRL/haim_drl/run_main_exp/HACO_Modified_230714-232414/HACO_Modified_HumanInTheLoopEnv_75d5a_00000_0_seed=0_2023-07-14_23-24-16'

    EPISODE_NUM_PER_CKPT = 1
    CKPT_START = 200
    CKPT_END = 201

    RENDER=True
    env_config = {
        "manual_control": True,
        "use_render": True,
        "controller": "keyboard",
        "window_size": (1600, 1100),
        "cos_similarity": True,
        "map": "CTO",
        "environment_num": 1,
        "start_seed": 15,
    }

    # initialize_ray(test_mode=False, local_mode=False, num_gpus=0)
    initialize_ray(test_mode=False, local_mode=False, num_gpus=1)

    def make_env(env_cfg=None):
        env_cfg = env_cfg or {}
        env_cfg.update(dict(manual_control=False, use_render=RENDER))
        return HumanInTheLoopEnv(env_cfg)


    from collections import defaultdict

    super_data = defaultdict(list)
    super_velocity_data = defaultdict(list)
    super_acceleration_data = defaultdict(list)

    env = make_env(env_config)

    # Create a CSV file and a writer
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # In your CSV header
        writer.writerow(["CKPT", "Success Rate", "Mean Episode Reward", "Mean Episode Cost",
                         "Mean Episode Disturbance Cost", "Mean Episode Disturbance Count",
                         "Mean Episode Disturbance Rate", "Mean Velocity", "Mean Acceleration"])

        # for ckpt_idx in range(CKPT_START, CKPT_END):
        for ckpt_idx in range(CKPT_START, CKPT_END, 10):
            ckpt = ckpt_idx

            compute_actions = get_function(CKPT_PATH, ckpt_idx)

            o = env.reset()
            epi_num = 0

            total_cost = 0
            total_reward = 0
            success_rate = 0
            ep_cost = 0
            ep_reward = 0
            success_flag = False
            horizon = 2000
            step = 0

            disturbance_count = 0  # New counter for disturbances
            total_disturbance_cost = 0
            ep_disturbance_cost = 0

            total_disturbance_count = 0
            total_disturbance_rate = 0

            velocity_list = []  # New list to store velocity data
            acceleration_list = []  # New list to store acceleration data
            previous_velocity = 0  # Placeholder for the velocity in the previous step

            while True:
                # action_to_send = compute_actions(w, [o], deterministic=False)[0]
                step += 1
                action_to_send = compute_actions(o)["default_policy"]
                o, r, d, info = env.step(action_to_send)


                # print(info["velocity"])

                total_reward += r
                ep_reward += r
                total_cost += info["cost"]
                ep_cost += info["cost"]

                total_disturbance_cost += info["disturbance_cost"]
                ep_disturbance_cost += info["disturbance_cost"]


                # Check if a disturbance occurred and increment the counter
                if info["disturbance"]:
                    disturbance_count += 1

                # Add the velocity to the velocity_list
                velocity_list.append(info["velocity"])

                # Calculate acceleration (current velocity - previous velocity )
                acceleration = info["velocity"] - previous_velocity
                acceleration_list.append(acceleration)

                # Save current velocity for the next step
                previous_velocity = info["velocity"]

                if d or step > horizon:
                    if info["arrive_dest"]:
                        success_rate += 1
                        success_flag = True
                    epi_num += 1
                    disturbance_rate = disturbance_count / step  # Calculate the ratio of disturbances

                    total_disturbance_count += disturbance_count
                    total_disturbance_rate += disturbance_rate

                    # Inside the episode loop
                    avg_velocity = sum(velocity_list) / step if step > 0 else 0
                    avg_acceleration = sum(acceleration_list) / step if step > 0 else 0

                    super_data[ckpt].append({
                        "reward": ep_reward,
                        "success": success_flag,
                        "cost": ep_cost,
                        "disturbance_cost": ep_disturbance_cost,
                        "disturbance_count": disturbance_count,
                        "disturbance_rate": disturbance_rate,
                        "avg_velocity": avg_velocity,
                        "avg_acceleration": avg_acceleration
                    })

                    super_velocity_data[ckpt].append({ "velocity": velocity_list * 10})
                    super_acceleration_data[ckpt].append({"acceleration": acceleration_list * 10})


                    ep_cost = 0.0
                    ep_reward = 0.0
                    ep_disturbance_cost = 0.0
                    success_flag = False
                    step = 0
                    disturbance_count = 0  # Reset the disturbance counter for the next episode

                    velocity_list = []  # Reset velocity list for next episode

                    if epi_num >= EPISODE_NUM_PER_CKPT:
                        break
                    else:
                        o = env.reset()

            mean_episode_success_rate = success_rate / EPISODE_NUM_PER_CKPT
            mean_episode_reward = total_reward / EPISODE_NUM_PER_CKPT
            mean_episode_cost = total_cost / EPISODE_NUM_PER_CKPT
            mean_episode_disturbance_cost = total_disturbance_cost / EPISODE_NUM_PER_CKPT
            mean_disturbance_count = total_disturbance_count / EPISODE_NUM_PER_CKPT
            mean_disturbance_rate = total_disturbance_rate / EPISODE_NUM_PER_CKPT

            # When writing data to CSV
            mean_velocity = sum([ep["avg_velocity"] for ep in super_data[ckpt]]) / EPISODE_NUM_PER_CKPT
            mean_acceleration = sum([ep["avg_acceleration"] for ep in super_data[ckpt]]) / EPISODE_NUM_PER_CKPT


            # Print to console
            print(
                "CKPT:{} | success_rate:{}, mean_episode_reward:{}, mean_episode_cost:{}, mean_episode_disturbance_cost:{}, mean_disturbance_count:{}, mean_disturbance_rate:{}, mean_velocity:{}, mean_acceleration:{}".format(
                    ckpt,
                    mean_episode_success_rate,
                    mean_episode_reward,
                    mean_episode_cost,
                    mean_episode_disturbance_cost,
                    mean_disturbance_count,
                    mean_disturbance_rate,
                    mean_velocity,
                    mean_acceleration
                ))

            # Write to CSV
            writer.writerow([ckpt, mean_episode_success_rate, mean_episode_reward, mean_episode_cost,
                             mean_episode_disturbance_cost, mean_disturbance_count, mean_disturbance_rate,
                             mean_velocity, mean_acceleration])

            del compute_actions

    env.close()

    import json

    # Saving super_data into a json file
    try:
        with open("eval_haim_drl_rej.json", "w") as f:
            json.dump(super_data, f)
    except Exception as e:
        print(f"An error occurred while saving super_data: {e}")

    print(super_data)

    # # Saving super_velocity_data into a json file
    # try:
    #     with open("eval_haco_velocity.json", "w") as f:
    #         json.dump(super_velocity_data, f)
    # except Exception as e:
    #     print(f"An error occurred while saving super_velocity_data: {e}")
    #
    # # print(super_velocity_data)
    #
    # try:
    #     with open("eval_haco_acceleration.json", "w") as f:
    #         json.dump(super_acceleration_data, f)
    # except Exception as e:
    #     print(f"An error occurred while saving super_acceleration_data: {e}")

    # print(super_acceleration_data)
