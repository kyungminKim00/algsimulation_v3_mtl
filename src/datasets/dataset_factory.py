"""
    A factory-pattern class which returns classification image/label pairs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import convert_snp_v1
from datasets import convert_snp_v2

datasets_map = {
    "snp_x0_20_y10_v1": convert_snp_v1,
    "snp_x0_20_y10_v2": convert_snp_v2,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """Given a dataset name and a split_name returns a Dataset.

    Args:
      name: String, the name of the dataset.
      split_name: A train/test split name.
      dataset_dir: The directory where the dataset files are stored.
      file_pattern: The file pattern to use for matching the dataset source files.
      reader: The subclass of tf.ReaderBase. If left as `None`, then the default
        reader defined by each dataset is used.

    Returns:
      A `Dataset` class.

    Raises:
      ValueError: If the dataset `name` is unknown.
    """
    if name not in datasets_map:
        raise ValueError("Name of dataset unknown %s" % name)
    return datasets_map[name].get_split(split_name, dataset_dir, file_pattern, reader)
