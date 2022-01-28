import os
import sys


def make_exists(storage_name):
    if not os.path.exists(storage_name):
        dir_name, file_name = os.path.split(os.path.abspath(sys.argv[0]))
        os.makedirs(os.path.join(dir_name, storage_name))


def load_txt_meta(path_to_meta):
    """load labels from txt
       :path_to_meta: path to meta data with txt format
       :return: labels: list of labels
    """
    labels = []
    with open(path_to_meta, 'r') as f:
        for line in f:
            labels.append(line.strip().split('\t')[-1])
    return labels


def write_to_txt(data, filename):
    """Save data with txt format in current dir
        :data: desired data, list
        :filename: str
        :return: Null
    """
    with open(filename, 'w') as f:
        for item in data:
            f.write(item + "\n")
