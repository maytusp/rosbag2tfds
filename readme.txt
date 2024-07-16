# Generate the dataset
tfds build



# Test loading the dataset
python -c "import tensorflow_datasets as tfds; ds = tfds.load('drone_nav', split='train'); print(ds)"
