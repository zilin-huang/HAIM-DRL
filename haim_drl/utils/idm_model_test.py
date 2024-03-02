"""
IDM Model

"""

import math
import numpy as np
import matplotlib.pyplot as plt


# Parameter Definitions
NUM = 6        # Number of vehicles
TIME = 300     # Number of time steps for simulation
LENGTH = 800    # Road length
DINIT = 15      # Initial distance between vehicles
DSAFE = 5       # Safe distance
TSAFE = 1       # Safe time headway
VINIT = 0      # Initial velocity of vehicles
# VMAX = 14       # Maximum velocity of vehicles
VMIN = 0        # Minimum velocity of vehicles
AINIT = 0       # Initial acceleration of vehicles
AMAX = 4        # Maximum acceleration of vehicles
AMIN = -4       # Minimum acceleration (comfortable deceleration) of vehicles
STEP = 0.1      # Time step size


x = np.zeros((NUM, TIME))
v = np.zeros((NUM, TIME))
a = np.zeros((NUM, TIME))


def initialize_vehicles():
    """
    Initialize vehicle data
    """

    for i in range(NUM):
        x[i][0] = (NUM - i - 1) * DINIT
        v[i][0] = VINIT
        a[i][0] = AINIT


def calculate_acceleration(i, t, VMAX):
    """
    Calculate acceleration based on IDM model

    Parameters:
    - i: Index of the vehicle
    - t: Time step (Note: In this example, the loop starts from 1, so usually use t-1)

    Returns:
    - acc: Acceleration
    """

    vt = v[i][t - 1]  # Velocity of the i-th vehicle at time step t
    vmax = VMAX  # Maximum velocity
    s0 = DSAFE  # Safe distance
    T = TSAFE  # Safe time headway
    alpha = AMAX  # Maximum acceleration
    beta = -AMIN  # Maximum deceleration (comfortable deceleration)
    if i == 0:
        dx = 99999  # Set dx to a large value 99999, indicating an infinite distance between the vehicle and the preceding vehicle
        dv = vt  # The velocity difference between the vehicle and the preceding vehicle is the current velocity
    else:
        dx = x[i - 1][t - 1] - x[i][t - 1]  # Distance between the vehicle and the preceding vehicle
        dv = v[i][t - 1] - v[i - 1][t - 1]  # Velocity difference between the vehicle and the preceding vehicle

    sn = s0 + vt * T + vt * dv / (2 * math.sqrt(alpha * beta))
    acc = alpha * (1 - pow(vt / vmax, 4) - pow(sn / dx, 2))
    return acc


def update_vehicles(i, t, brake_time, brake_acc, decelerate_time, decelerate_acc, VMAX):
    """
    Update vehicle data

    Parameters:
    - i: Index of the vehicle
    - t: Time step (Note: In this example, the loop starts from 1, so usually use t-1)
    - brake_time: Time range for braking
    - brake_acc: Braking deceleration
    - decelerate_time: Time range for deceleration
    - decelerate_acc: Deceleration value for deceleration
    - VMAX: Maximum velocity

    Returns:
    - a: Acceleration
    - v: Velocity
    - x: Position
    """
    if i == 0:
        if t >= brake_time[0] and t <= brake_time[1]:
            # Scenario for strong braking of the first vehicle
            a[i][t] = brake_acc
            v[i][t] = v[i][t - 1] + a[i][t] * STEP
        # elif t >= decelerate_time[0] and t <= decelerate_time[1]:
        #     # Scenario for slow braking of the first vehicle
        #     a[i][t] = decelerate_acc
        #     v[i][t] = v[i][t - 1] + a[i][t] * STEP
        else:
            # Scenario for free flow of the first vehicle
            a[i][t] = calculate_acceleration(i, t, VMAX)
            v[i][t] = v[i][t - 1] + a[i][t] * STEP
    else:
        a[i][t] = calculate_acceleration(i, t, VMAX)  # calculate_acceleration function automatically subtracts 1, i.e., the acceleration at time t is calculated using the velocity at time t-1
        v[i][t] = v[i][t - 1] + a[i][t] * STEP  # Velocity at time t is equal to the velocity at time t-1 plus at

    x[i][t] = x[i][t - 1] + v[i][t - 1] * STEP + 0.5 * a[i][t] * STEP * STEP    # Position at time t is equal to the position at time t-1 plus (v*t + 1/2*a*t^2)

    if a[i][t] > AMAX:
        a[i][t] = AMAX
    if a[i][t] < AMIN:
        a[i][t] = AMIN
    if v[i][t] > VMAX:
        v[i][t] = VMAX
    if v[i][t] < VMIN:
        v[i][t] = VMIN


def plot_result():
    """
    Plot: position
    """
    plt.figure()
    i = range(TIME)
    for n in range(NUM):
        plt.plot(i, x[n], label=f"Vehicle {n + 1}")

    # You can adjust these parameters to increase the font size.
    plt.xlabel('$time(s)$', fontsize=18)
    plt.ylabel('$position(m)$', fontsize=18)
    # plt.title('Vehicle Positions Over Time', fontsize=16)
    plt.xlim(0, 100)  # Set x-axis limit to 0-100
    plt.legend(fontsize=16)
    plt.savefig('idm_position_only.png')  # save figure to local file
    plt.show()





def simulate(brake_time, brake_acc, decelerate_time, decelerate_acc, VMAX):
    """
    Start simulating the simulation
    """
    initialize_vehicles()

    for t in np.arange(1, TIME):
        for i in range(NUM):
            update_vehicles(i, t, brake_time, brake_acc, decelerate_time, decelerate_acc, VMAX)    # Update a, v, and s

    plot_result()


def get_vehicle_data(brake_time, brake_acc, decelerate_time, decelerate_acc, VMAX):
    """
    Get the velocity, acceleration, and position data for the specified vehicle

    Parameters:
    - vehicle_index: Vehicle index (starting from 0)

    Returns:
    - velocity: List of velocity data
    - acceleration: List of acceleration data
    - position: List of position data
    """

    # Run the IDM model simulation
    simulate(brake_time, brake_acc, decelerate_time, decelerate_acc, VMAX)

    velocity = v
    acceleration = a
    position = x
    return velocity, acceleration, position


if __name__ == "__main__":

    # Maximum velocity of vehicles
    VMAX = 15

    # Set brake time and brake acceleration
    brake_time = (100, 150)
    brake_acc = -4

    decelerate_time = (70, 80)
    decelerate_acc = 0

    # Run the IDM model simulation
    velocities, accelerations, positions = get_vehicle_data(brake_time, brake_acc, decelerate_time, decelerate_acc, VMAX)

    # IDM velocities of the following vehicles (excluding the first vehicle)
    velocities_limited = velocities[1:, :]

    # Average velocity of the following vehicles (excluding the first vehicle)
    average_velocity = np.mean(velocities_limited)
    print(average_velocity)
