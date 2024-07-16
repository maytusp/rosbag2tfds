import rosbag
import cv2
import numpy as np
import os
import tqdm
from scipy import signal

np.random.seed(42)

bag_dir = 'bag'
train_path = 'data/drone_pointnav_it/train'
val_path = 'data/drone_pointnav_it/val'
train_ratio = 0.9

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
filename_list = os.listdir(bag_dir)
NUM_EPS = len(filename_list)
ORIG_ALL_EPS = [ep for ep in range(1, NUM_EPS+1)]
NUM_TRAIN_EPS = int(train_ratio*NUM_EPS)
NUM_VAL_EPS = NUM_EPS - NUM_TRAIN_EPS
ALL_EPS = np.random.choice(ORIG_ALL_EPS, NUM_EPS, replace=False)

VAL_EPS  = set(ALL_EPS[:NUM_VAL_EPS])
TRAIN_EPS = set(ALL_EPS[NUM_VAL_EPS:])
total_dur = 0

def state_diff(current_state, next_state, axis=3):
    '''
    state: [X,Y,Z,YAW]
    '''
    diff = next_state - current_state
    yaw_diff = diff[axis]

    new_yaw_diff = np.copy(yaw_diff)
    # If the difference is greater than pi, adjust by subtracting 2*pi
    if yaw_diff > np.pi:
        new_yaw_diff -= 2 * np.pi
    
    # If the difference is less than -pi, adjust by adding 2*pi
    elif yaw_diff < -np.pi:
        new_yaw_diff += 2 * np.pi
    cont_diff = np.concatenate((diff[:axis], np.array([new_yaw_diff])))
    return cont_diff


def create_episode(path, state_array, obs_array):
    episode = []
    episode_length = len(obs_array)
    assert len(state_array) == episode_length
    for step in range(episode_length):
        terminal_step = episode_length - 1
        goal_array = state_array[terminal_step] - state_array[step]
        goal_array = goal_array[:2] # Goal is only (x, y) position
        is_terminal = False
        if step == terminal_step:
            action = np.zeros_like(state_diff(state_array[step], state_array[step]))
            is_terminal = True
        else:
            action = state_diff(state_array[step], state_array[step+1])
        episode.append({
            'image': obs_array[step],
            'state': state_array[step],
            'action': action,
            'goal': goal_array,
            'is_terminal' : is_terminal
        })
    np.save(path, episode)

for ep_id in tqdm.tqdm(range(1,NUM_EPS+1)):
    # Path to the ROS bag file
    bag_file = os.path.join(bag_dir, f'ep{ep_id}.bag')

    # Topic name of the compressed image
    topic = '/data_out'
    x_arr = []
    y_arr = []
    z_arr = []
    yaw_arr = []
    image_arr = []
    # Open the bag file
    down_sample_factor = 6 # Downsample from 30Hz to 5Hz
    step = 0
    x_prev = 0
    x_curr = 0
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            header = msg.header
            x_curr = msg.x_curr
            y_curr = msg.y_curr
            z_curr = msg.z_curr
            yaw_curr = msg.yaw_curr
            image_data = np.frombuffer(msg.image.data, np.uint8)
            cv_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR) # BGR image
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            if step % down_sample_factor == 0:
                # x_arr.append(x_curr)
                # y_arr.append(y_curr)
                # z_arr.append(z_curr)
                # yaw_arr.append(yaw_curr)
                image_arr.append(cv_image_rgb)
                total_dur+=1

            # New method: LPF + Downsamplinng
            x_arr.append(x_curr)
            y_arr.append(y_curr)
            z_arr.append(z_curr)
            yaw_arr.append(yaw_curr)

            # if step > 0:
            #     if (x_curr-x_prev > 0.02 or x_curr-x_prev < -0.02):
            #         print(f"frame {step} has exceed velocity {x_curr-x_prev}")
            # x_prev = x_curr
            # y_prev = y_curr

            step+=1
        x_arr = signal.decimate(x_arr, down_sample_factor)
        y_arr = signal.decimate(y_arr, down_sample_factor)
        z_arr = signal.decimate(z_arr, down_sample_factor)
        yaw_arr = signal.decimate(yaw_arr, down_sample_factor)
        state_array = np.array([x_arr, y_arr, z_arr, yaw_arr]) # (4, T)
        state_array = np.transpose(state_array, (1,0)) # (T, 4)
        obs_array = np.array(image_arr)
        if len(state_array) != len(obs_array):
            print(len(state_array), len(obs_array))
        # Sanity Check
        # print(state_array.shape)
        # print(obs_array.shape)
        # If ep id is in train id
        if ep_id in TRAIN_EPS:
            path = train_path
        elif ep_id in VAL_EPS:
            path = val_path
        else:
            print(f"No episode id {ep_id}")
            raise TypeError("Episode id is wrong")
        path = os.path.join(path, f"ep{ep_id}.npy")

        create_episode(path, state_array, obs_array)
print(f"TOTAL FRAMES: {total_dur}")