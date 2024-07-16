from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

def resize_and_crop(frame, min_size=256):
    '''
    frame has the size of (H, W, 3)
    '''
    # Calculate the new dimensions maintaining the aspect ratio
    if frame.shape[2] == 3:
        height, width, channel = frame.shape
    else:
        channel, height, width = frame.shape

    if width < height:
        new_width = min_size
        new_height = int(height * (min_size / width))
    else:
        new_height = min_size
        new_width = int(width * (min_size / height))
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Perform center crop
    center_x, center_y = new_width // 2, new_height // 2
    crop_size = min_size
    if frame.shape[2] == 3:
        cropped_frame = resized_frame[
            center_y - crop_size // 2 : center_y + crop_size // 2,
            center_x - crop_size // 2 : center_x + crop_size // 2
        ]
    else:
        cropped_frame = resized_frame[
            :,
            center_y - crop_size // 2 : center_y + crop_size // 2,
            center_x - crop_size // 2 : center_x + crop_size // 2
        ]        
    return cropped_frame

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='camera image.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float32,
                            doc='absolute x y z yaw (world coordinate)',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(4,),
                        dtype=np.float32,
                        doc='dx dy dz dyaw',
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'goal': tfds.features.Tensor(
                        shape=(2,),
                        dtype=np.float32,
                        doc='point goal'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/home/maytusp/Projects/drone_tta_rosbag/data/drone_pointnav_it/train/*.npy'),
            'val': self._generate_examples(path='/home/maytusp/Projects/drone_tta_rosbag/data/drone_pointnav_it/val/*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # print("episode_path", episode_path)
            # episode_path = tf.io.gfile.GFile(episode_path, mode='r')
            # load raw data --> this should change for your dataset
            
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                action = step['action'].astype(np.float32)
                state = step['state'].astype(np.float32)
                goal = step['goal'].astype(np.float32)
                is_terminal = step['is_terminal']
                cropped_frame = resize_and_crop(step['image'])
                episode.append({
                    'observation': {
                        'image': cropped_frame,
                        'state': state,
                    },
                    'action': action,
                    'is_terminal': is_terminal,
                    'goal': goal,
                })
                action_prev = action
            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)
        # print("episode_paths", episode_paths)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            # print("sample", sample)
            yield _parse_example(sample)