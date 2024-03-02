from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# Initialize an EventAccumulator and load the event file
event_acc = EventAccumulator('/home/zilin/code/HAIM-DRL/haim_drl/run_main_exp/HACO_Modified_230714-232414/HACO_Modified_HumanInTheLoopEnv_75d5a_00000_0_seed=0_2023-07-14_23-24-16/events.out.tfevents.1689395056.zilin-Legion-5-Pro-16ACH6H')
event_acc.Reload()

# Extract the data for the desired metric
# Replace 'your_metric_name' with the name of the metric you want to plot
events = event_acc.Scalars('ray/tune/custom_metrics/total_takeover_cost_mean')

# Split the events into steps and values
steps = [e.step for e in events]
values = [e.value for e in events]

# Plot the data
plt.plot(steps, values)
plt.xlabel('Step')
plt.ylabel('total_takeover_cost_mean')
plt.title('total_takeover_cost_mean over Steps')
plt.show()
