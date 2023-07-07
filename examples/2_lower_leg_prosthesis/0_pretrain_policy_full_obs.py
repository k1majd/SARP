import os
from tensorflow import keras
import tensorflow as tf
import csv
import numpy as np
import pandas as pd
from sarp.utils import (
    load_expert_data_hospital,
    separate_train_test,
    curriculum_training,
)

if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.realpath(__file__)) + f"/data/expert_data/sample2.csv"
    # with open(data_dir) as csv_file:
    #     data = np.asarray(
    #         list(csv.reader(csv_file, delimiter=",")), dtype=np.float32
    #     )
    
    

    data = pd.read_csv(data_dir)
    print(data.shape)