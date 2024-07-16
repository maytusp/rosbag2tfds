"""drone_pointnav dataset."""

import drone_pointnav_dataset_builder
import tensorflow_datasets as tfds

class DronePointnavTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for drone_pointnav dataset."""
  # TODO(drone_pointnav):
  DATASET_CLASS = drone_pointnav_dataset_builder.Builder
  SPLITS = {
      'train': 90,  # Number of fake train example
      'test': 10,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
