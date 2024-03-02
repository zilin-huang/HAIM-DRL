import copy

import numpy as np
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.engine.engine_utils import get_global_config
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.manual_control_policy import TakeoverPolicy
from metadrive.utils.math_utils import safe_clip

import haim_drl.utils.idm_model as idm_model
import math

import csv

ScreenMessage.SCALE = 0.1

brake_value = -5  # The deceleration threshold that triggers the calculation of traffic disturbance costs

class MyKeyboardController(KeyboardController):
    # Update Parameters
    STEERING_INCREMENT = 0.05
    STEERING_DECAY = 0.5

    THROTTLE_INCREMENT = 0.5
    THROTTLE_DECAY = 1

    BRAKE_INCREMENT = 0.5
    BRAKE_DECAY = 1

class MyTakeoverPolicy(TakeoverPolicy):
    """
    Record the takeover signal
    """
    def __init__(self, obj, seed):
        super(TakeoverPolicy, self).__init__(obj, seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "joystick":
                self.controller = SteeringWheelController()
            elif config["controller"] == "keyboard":
                self.controller = MyKeyboardController(False)
            else:
                raise ValueError("Unknown Policy: {}".format(config["controller"]))
        self.takeover = False


class HumanInTheLoopEnv(SafeMetaDriveEnv):
    """
    This Env depends on the new version of MetaDrive
    """

    def default_config(self):
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(
            {
                "environment_num": 50,
                "start_seed": 100,
                "cost_to_reward": True,
                "traffic_density": 0.2,    # Twice the raw parameter (0.06)
                "manual_control": False,
                "controller": "joystick",
                "agent_policy": MyTakeoverPolicy,
                "only_takeover_start_cost": True,
                "main_exp": True,
                "random_spawn": False,
                "cos_similarity": True,
                "out_of_route_done": True,
                "in_replay": False,
                "random_spawn_lane_index": False,

                "traffic_disturbance_start": False,     # Traffic disturbance parameters
                "only_traffic_disturbance_start_cost": True,    # Traffic disturbance parameters

                "target_vehicle_configs": {
                    "default_agent": {"spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 1)}}
            },
            allow_add_new_key=True
        )
        return config

    def reset(self, *args, **kwargs):
        self.in_stop = False
        self.t_o = False
        self.total_takeover_cost = 0
        self.input_action = None

        self.newbie_acceleration = 0    # throttle of agent
        self.newbie_steering = 0    # steering of agent

        self.last_velocity = 0  # Initialize the speed of the previous frame to 0

        self.t_disturbance = False  # Similar to self.t_o
        self.total_disturbance_cost = 0

        self.position = 0  # Initialize position at the start of each episode

        ret = super(HumanInTheLoopEnv, self).reset(*args, **kwargs)
        if self.config["random_spawn"]:
            self.config["vehicle_config"]["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2,
                                                                 self.engine.np_random.randint(3))
        # keyboard is not as good as steering wheel, so set a small speed limit
        self.vehicle.update_config({"max_speed": 25 if self.config["controller"] == "keyboard" else 40})
        return ret

    def _get_step_return(self, actions, engine_info):
        o, r, d, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        # print("o:", len(o))

        if self.config["in_replay"]:
            return o, r, d, engine_info

        controller = self.engine.get_policy(self.vehicle.id)
        last_t = self.t_o
        self.t_o = controller.takeover if hasattr(controller, "takeover") else False
        engine_info["takeover_start"] = True if not last_t and self.t_o else False
        # not last_t determines whether the value of the variable last_t is False
        # not last_t and self.t_o, if both conditions are true, returns True, otherwise returns False (i.e., False-True)
        engine_info["takeover"] = self.t_o
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.t_o

        if not condition:
            self.total_takeover_cost += 0
            engine_info["takeover_cost"] = 0
        else:
            # In the case of a takeover, for the first action
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost

        # Add: newbie information of acceleration/steering to info
        self.newbie_acceleration = self.get_newbie_acceleration(engine_info)
        engine_info["newbie_acceleration"] = self.newbie_acceleration
        self.newbie_steering = self.get_newbie_steering(engine_info)
        engine_info["newbie_steering"] = self.newbie_steering

        # Calculate the acceleration of the vehicle
        current_velocity = engine_info["velocity"] * (1 / 3.6)  # The initial speed is km/h, which is converted to m/s
        velocity_difference = (current_velocity - self.last_velocity) * 10  # Since the frame update speed is 0.1s, multiply by 10
        self.last_velocity = current_velocity

        # When the brake deceleration exceeds a brake_value and not take over
        last_t_d = self.t_disturbance
        self.t_disturbance = True if velocity_difference < brake_value and not self.t_o else False   # If it less than -5 and not take over
        engine_info["traffic_disturbance_start"] = True if not last_t_d and self.t_disturbance else False   # If false-true
        engine_info["disturbance"] = self.t_disturbance
        disturbance_condition = engine_info["traffic_disturbance_start"] if self.config["only_traffic_disturbance_start_cost"] else self.t_disturbance  # Find out the first action

        time_step = 0.1  # 假设每一步的时间为0.1秒
        self.position += self.last_velocity * time_step  # last_velocity is in m/s
        engine_info["current_position"] = self.position

        if not disturbance_condition:
            self.disturbance_cost = 0
            self.total_disturbance_cost += 0
            engine_info["disturbance_cost"] = 0
        else:
            # In the case of traffic disturbance caused by braking, for the first action
            self.disturbance_cost = self.get_traffic_disturbance_cost(velocity_difference, engine_info)
            self.total_disturbance_cost += self.disturbance_cost
            engine_info["disturbance_cost"] = self.disturbance_cost

        engine_info["total_disturbance_cost"] = self.total_disturbance_cost     # add total_disturbance_cost
        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["total_native_cost"] = self.episode_cost

        # print(engine_info)

        return o, r, d, engine_info

    def _is_out_of_road(self, vehicle):
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        self.input_action = copy.copy(actions)
        ret = super(HumanInTheLoopEnv, self).step(actions)
        while self.in_stop:
            self.engine.taskMgr.step()
        if self.config["use_render"] and self.config["main_exp"] and not self.config["in_replay"]:
            super(HumanInTheLoopEnv, self).render(text={
                "Total Cost": self.episode_cost,
                "Takeover Cost": self.total_takeover_cost,
                "Takeover": self.t_o,
                "Disturbance": self.t_disturbance,
                "COST": ret[-1]["takeover_cost"],
                "Velocity": ret[-1]["velocity"],
                # "Stop (Press E)": ""
                "D Cost": ret[-1]["disturbance_cost"],
                "Disturbance Cost": self.total_disturbance_cost
            })

        return ret

    def stop(self):
        self.in_stop = not self.in_stop

    def setup_engine(self):
        super(HumanInTheLoopEnv, self).setup_engine()
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        if not self.config["cos_similarity"]:
            return 1
        takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.input_action), -1, 1)
        # cos_dist = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1]) / 1e-6 +(
        #         np.linalg.norm(takeover_action) * np.linalg.norm(agent_action))

        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident

        return 1 - cos_dist

    def get_newbie_acceleration(self, info):
        # takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.input_action), -1, 1)
        return agent_action[1]

    def get_newbie_steering(self, info):
        # takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.input_action), -1, 1)
        return agent_action[0]


    def get_traffic_disturbance_cost(self, velocity_difference, info):
        """
        Calculate disturbance cost
        """
        # Set to the desire speed of the IDM model
        VMAX = info["velocity"]

        # Set brake time and brake acceleration
        brake_time = (100, 150)
        brake_acc_1 = velocity_difference   # brake acceleration

        decelerate_time = (70, 80)
        decelerate_acc = 0

        # Run simulation IDM model
        velocities_1, accelerations_1, positions_1 = idm_model.get_vehicle_data(brake_time, brake_acc_1, decelerate_time, decelerate_acc, VMAX)

        # IDM speed/acceleration of the vehicle behind (not counting the first vehicle)
        velocities_limited_1 = velocities_1[1:, :]
        accelerations_limited_1 = accelerations_1[1:, :]

        # Average speed of the vehicle behind (not counting the first vehicle)
        average_velocity_1 = np.mean(velocities_limited_1)
        average_acceleration_1 = np.mean(accelerations_limited_1)
        # print("Brake velocity", average_velocity_1)
        # print("Brake acceleration", average_acceleration_1)

        """
        Ideal case
        """
        brake_acc_2 = 0
        velocities_2, accelerations_2, positions_2 = idm_model.get_vehicle_data(brake_time, brake_acc_2, decelerate_time, decelerate_acc, VMAX)
        velocities_limited_2 = velocities_2[1:, :]
        accelerations_limited_2 = accelerations_2[1:, :]
        average_velocity_2 = np.mean(velocities_limited_2)
        average_acceleration_2 = np.mean(accelerations_limited_2)
        # print("Ideal velocity", average_velocity_2)
        # print("Ideal acceleration", average_acceleration_2)

        # Difference in average speed
        average_velocity_difference = average_velocity_2 - average_velocity_1

        return 1 - math.exp(-average_velocity_difference)    # Take its exponent, as the cost



# if __name__ == "__main__":
#     env = HumanInTheLoopEnv(
#       # {"manual_control": True, "disable_model_compression": True, "use_render": True, "main_exp": True})
#         {"manual_control": True, "disable_model_compression": True, "use_render": True, "main_exp": True, "controller": "keyboard"})
#
#     env.reset()
#     while True:
#         env.step([0, 0])


if __name__ == "__main__":
    env = HumanInTheLoopEnv(
        {"manual_control": True, "disable_model_compression": True, "use_render": True, "main_exp": True,
         "controller": "keyboard"}
    )

    # Open a CSV file to write the velocity and position in the current directory
    with open('vehicle_velocity_and_position_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Velocity', 'Position'])  # Write headers to the CSV file

        env.reset()
        current_time = 0  # Initialize a variable to keep track of the simulation time

        while True:
            o, r, d, engine_info = env.step([0, 0])
            # Save the current velocity and position to the CSV file
            writer.writerow([current_time, engine_info["velocity"], engine_info["current_position"]])
            current_time += 0.1  # Assuming each step in the simulation is 0.1 seconds

            if d:  # If the environment returns 'done', break out of the loop
                break
