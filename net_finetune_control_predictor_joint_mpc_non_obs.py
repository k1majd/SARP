import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import tensorboard
import tensorflow as tf
import pandas as pd

# import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from shapely.geometry import Polygon
from shapely.affinity import scale
from datetime import datetime
from pyomo.gdp import *
from scipy.spatial import ConvexHull

from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import ConstraintsClass
from nnreplayer.repair.repair_weights_class import NNRepair


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(
            list(csv.reader(csv_file, delimiter=",")), dtype=np.float32
        )
    return data


def squared_sum(x, y):
    m, n = np.array(x).shape
    _squared_sum = 0
    for i in range(m):
        for j in range(n):
            _squared_sum += (x[i, j] - y[i, j]) ** 2
    return _squared_sum


def generateDataWindow(window_size, ctrl_horizon):
    Dfem = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_1.csv"
    )  # femur time, angle, velocity
    Dtib = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_2.csv"
    )  # shin time, angle, velocity
    Dfut = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_3.csv"
    )  # foot time, angle, velocity
    n = 20364
    Dankle = np.subtract(Dtib[: n, 1], Dfut[:n, 1])
    Dankle = (Dankle - Dankle.mean(0))/Dankle.std(0)  # normalize Dankle
    observations = np.concatenate((Dfem[:n, 1:], Dtib[:n, 1:]), axis=1)
    observations = (observations - observations.mean(0)) / observations.std(0)
    observations = np.concatenate(
        (
            observations,
            Dankle[:n].reshape(n, 1),
        ),
        axis=1,
    )
    controls = Dankle  # (Dankle - Dankle.mean(0))/Dankle.std(0)
    n_train = 18188
    # n_train = 500
    # train dataset
    train_observation = np.array([]).reshape(0, 5 * window_size)
    for i in range(n_train):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        train_observation = np.concatenate(
            (train_observation, temp_obs), axis=0
        )
    train_controls = np.array([]).reshape(0, ctrl_horizon)
    train_output = np.array([]).reshape(0, 4 * ctrl_horizon)
    for i in range(n_train):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, controls[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observations[i + window_size + j, :-1].reshape(1, -1),
                ),
                axis=1,
            )
        train_controls = np.concatenate((train_controls, temp_ctrl), axis=0)
        train_output = np.concatenate((train_output, temp_output), axis=0)
    # train_controls = controls[window_size:n_train + window_size].reshape(
    #     -1, 1
    # )
    # train_output = observations[window_size:n_train + window_size, :-1]

    # test dataset
    test_observation = np.array([]).reshape(0, 5 * window_size)
    for i in range(n_train, n - window_size - ctrl_horizon):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        test_observation = np.concatenate((test_observation, temp_obs), axis=0)
    test_controls = np.array([]).reshape(0, ctrl_horizon)
    test_output = np.array([]).reshape(0, 4 * ctrl_horizon)
    for i in range(n_train, n - window_size - ctrl_horizon):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, controls[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observations[i + window_size + j, :-1].reshape(1, -1),
                ),
                axis=1,
            )
        test_controls = np.concatenate((test_controls, temp_ctrl), axis=0)
        test_output = np.concatenate((test_output, temp_output), axis=0)
    # test_controls = controls[n_train + window_size :].reshape(-1, 1)
    # test_output = observations[n_train + window_size :, :-1]
    return (
        train_observation[238:],
        np.concatenate((train_controls[238:], train_output[238:]), axis=1),
        test_observation,
        np.concatenate((test_controls, test_output), axis=1),
    )


def generateDataWindowForce2(window_size, ctrl_horizon):
    pad = window_size + ctrl_horizon
    # walking_ranges = np.array([[600, 2760-pad], [3500, 6045-pad], [7000, 9500-pad]])
    walking_ranges = np.array([[7185, 9000-pad], [9995, 11150-pad], [11500, 12530-pad]])
    # walking_ranges = np.array([[33270, 34955-pad], [37765, 39340-pad], [40270, 42000-pad]])
    w_d = loadData("/Users/keyvanmajd/state_control_repair/demos/examples/tc7_walking_example/repair_bound_constraint/data/data_full.csv") 
    Dankle = w_d[:,37] # ankle angle
    Dankle = Dankle.reshape((Dankle.shape[0], 1))
    observation = w_d[:,[14, 18, 7, 11]] # femur angle, vel, tibia angle, vel
    # observation = (observation - observation.mean(0)) / observation.std(0)
    print(f"observation mean of train data = {observation[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)}, std = {observation[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)}")
    print(f"observation mean of validation data = {observation[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)}, std = {observation[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)}")
    print(f"observation mean of test data = {observation[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)}, std = {observation[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)}")
    observation[
        walking_ranges[0][0]:walking_ranges[0][1]+pad
        ] = (
            observation[walking_ranges[0][0]:walking_ranges[0][1]+pad]
            - observation[walking_ranges[0][0]:walking_ranges[0][1]+pad].mean(0)
            )/observation[walking_ranges[0][0]:walking_ranges[0][1]+pad].std(0)
    observation[
        walking_ranges[1][0]:walking_ranges[1][1]+pad
        ] = (
            observation[walking_ranges[1][0]:walking_ranges[1][1]+pad]
            - observation[walking_ranges[1][0]:walking_ranges[1][1]+pad].mean(0)
            )/observation[walking_ranges[1][0]:walking_ranges[1][1]+pad].std(0)
    observation[
        walking_ranges[2][0]:walking_ranges[2][1]+pad
        ] = (
            observation[walking_ranges[2][0]:walking_ranges[2][1]+pad]
            - observation[walking_ranges[2][0]:walking_ranges[2][1]+pad].mean(0)
            )/observation[walking_ranges[2][0]:walking_ranges[2][1]+pad].std(0)
#     observation = np.concatenate(
#         (
#             observation,
#             Dankle,
#         ),
#         axis=1,
#     )
    # force_meas = w_d[["insole_sensor_1", "insole_sensor_ 2", "insole_sensor_3", "insole_sensor_4", "insole_sensor_5", "insole_sensor_6", "insole_sensor_7", "insole_sensor_8", "insole_sensor_9", "insole_sensor_10", "insole_sensor_11", "insole_sensor_12", "insole_sensor_13", "insole_sensor_14", "insole_sensor_15", "insole_sensor_16"]].to_numpy()
    force_meas = w_d[:, [21, 22, 23, 24]] # insole sensors 1, 2, 3, and 4
    print(f"force_meas mean of train data = {force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)}, std = {force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)}")
    print(f"force_meas mean of validation data = {force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)}, std = {force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)}")
    print(f"force_meas mean of test data = {force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)}, std = {force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)}")
    force_meas[
        walking_ranges[0][0]:walking_ranges[0][1]+pad
        ] = (
            force_meas[walking_ranges[0][0]:walking_ranges[0][1]+pad]
            - force_meas[walking_ranges[0][0]:walking_ranges[0][1]+pad].mean(0)
            )/force_meas[walking_ranges[0][0]:walking_ranges[0][1]+pad].std(0)
    force_meas[
        walking_ranges[1][0]:walking_ranges[1][1]+pad
        ] = (
            force_meas[walking_ranges[1][0]:walking_ranges[1][1]+pad]
            - force_meas[walking_ranges[1][0]:walking_ranges[1][1]+pad].mean(0)
            )/force_meas[walking_ranges[1][0]:walking_ranges[1][1]+pad].std(0)
    force_meas[
        walking_ranges[2][0]:walking_ranges[2][1]+pad
        ] = (
            force_meas[walking_ranges[2][0]:walking_ranges[2][1]+pad]
            - force_meas[walking_ranges[2][0]:walking_ranges[2][1]+pad].mean(0)
            )/force_meas[walking_ranges[2][0]:walking_ranges[2][1]+pad].std(0)
    # force_meas = (force_meas - force_meas.mean(0)) / force_meas.std(0)

    # create training dataset
    obs_train = np.array([]).reshape(0, 4 * window_size)
    for i in range(walking_ranges[0][0], walking_ranges[0][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_train = np.concatenate(
            (obs_train, temp_obs), axis=0
        )

    ctrl_train = np.array([]).reshape(0, ctrl_horizon)
    out_train = np.array([]).reshape(0, (4 + force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[0][0], walking_ranges[0][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_train = np.concatenate((ctrl_train, temp_ctrl), axis=0)
        out_train = np.concatenate((out_train, temp_output), axis=0)

    # create validation dataset
    obs_val = np.array([]).reshape(0, 4 * window_size)
    for i in range(walking_ranges[1][0], walking_ranges[1][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_val = np.concatenate(
            (obs_val, temp_obs), axis=0
        )

    ctrl_val = np.array([]).reshape(0, ctrl_horizon)
    out_val = np.array([]).reshape(0, (4+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[1][0], walking_ranges[1][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_val = np.concatenate((ctrl_val, temp_ctrl), axis=0)
        out_val = np.concatenate((out_val, temp_output), axis=0)

    # create test dataset
    obs_test = np.array([]).reshape(0, 4 * window_size)
    for i in range(walking_ranges[2][0], walking_ranges[2][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_test = np.concatenate(
            (obs_test, temp_obs), axis=0
        )

    ctrl_test = np.array([]).reshape(0, ctrl_horizon)
    out_test = np.array([]).reshape(0, (4+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[2][0], walking_ranges[2][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_test = np.concatenate((ctrl_test, temp_ctrl), axis=0)
        out_test = np.concatenate((out_test, temp_output), axis=0)

    return (
        obs_train,
        ctrl_train,
        out_train,
        obs_val,
        ctrl_val,
        out_val,
        obs_test,
        ctrl_test,
        out_test,
    )


def generateDataWindowForce3(window_size, ctrl_horizon):
    # obs = [femur angle, vel, tibia angle, vel] * window_size
    # ctrl = [ankle angle] * ctrl_horizon
    # output = [femur angle, vel, tibia angle, vel, force] * ctrl_horizon
    pad = window_size + ctrl_horizon
    walking_ranges = np.array([[7185, 9000-pad], [9995, 11150-pad], [11500, 12530-pad]])
    w_d = loadData(os.path.dirname(os.path.realpath(__file__)) + "/data/data_full.csv")
    Dankle = w_d[:, 37]  # ankle angle
    Dankle = Dankle.reshape((Dankle.shape[0], 1))
    observation = w_d[:, [7, 11]]  # femur angle, vel, tibia angle, vel
    # observation = (observation - observation.mean(0)) / observation.std(0)
    print(f"observation mean of train data = {observation[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)}, std = {observation[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)}")
    print(f"observation mean of validation data = {observation[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)}, std = {observation[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)}")
    print(f"observation mean of test data = {observation[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)}, std = {observation[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)}")
    observation[
        walking_ranges[0][0]:walking_ranges[0][1]+pad
        ] = (
            observation[walking_ranges[0][0]:walking_ranges[0][1]+pad]
            - observation[walking_ranges[0][0]:walking_ranges[0][1]+pad].mean(0)
            )/observation[walking_ranges[0][0]:walking_ranges[0][1]+pad].std(0)
    observation[
        walking_ranges[1][0]:walking_ranges[1][1]+pad
        ] = (
            observation[walking_ranges[1][0]:walking_ranges[1][1]+pad]
            - observation[walking_ranges[1][0]:walking_ranges[1][1]+pad].mean(0)
            )/observation[walking_ranges[1][0]:walking_ranges[1][1]+pad].std(0)
    observation[
        walking_ranges[2][0]:walking_ranges[2][1]+pad
        ] = (
            observation[walking_ranges[2][0]:walking_ranges[2][1]+pad]
            - observation[walking_ranges[2][0]:walking_ranges[2][1]+pad].mean(0)
            )/observation[walking_ranges[2][0]:walking_ranges[2][1]+pad].std(0)
#     observation = np.concatenate(
#         (
#             observation,
#             Dankle,
#         ),
#         axis=1,
#     )
    # force_meas = w_d[["insole_sensor_1", "insole_sensor_ 2", "insole_sensor_3", "insole_sensor_4", "insole_sensor_5", "insole_sensor_6", "insole_sensor_7", "insole_sensor_8", "insole_sensor_9", "insole_sensor_10", "insole_sensor_11", "insole_sensor_12", "insole_sensor_13", "insole_sensor_14", "insole_sensor_15", "insole_sensor_16"]].to_numpy()
    force_meas = w_d[:, [21, 22, 23, 24]]  # insole sensors 1, 2, 3, and 4
    print(f"force_meas mean of train data = {force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)}, std = {force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)}")
    print(f"force_meas mean of validation data = {force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)}, std = {force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)}")
    print(f"force_meas mean of test data = {force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)}, std = {force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)}")
    force_meas[
        walking_ranges[0][0]:walking_ranges[0][1]+pad
        ] = (
            force_meas[walking_ranges[0][0]:walking_ranges[0][1]+pad]
            - force_meas[walking_ranges[0][0]:walking_ranges[0][1]+pad].mean(0)
            )/force_meas[walking_ranges[0][0]:walking_ranges[0][1]+pad].std(0)
    force_meas[
        walking_ranges[1][0]:walking_ranges[1][1]+pad
        ] = (
            force_meas[walking_ranges[1][0]:walking_ranges[1][1]+pad]
            - force_meas[walking_ranges[1][0]:walking_ranges[1][1]+pad].mean(0)
            )/force_meas[walking_ranges[1][0]:walking_ranges[1][1]+pad].std(0)
    force_meas[
        walking_ranges[2][0]:walking_ranges[2][1]+pad
        ] = (
            force_meas[walking_ranges[2][0]:walking_ranges[2][1]+pad]
            - force_meas[walking_ranges[2][0]:walking_ranges[2][1]+pad].mean(0)
            )/force_meas[walking_ranges[2][0]:walking_ranges[2][1]+pad].std(0)
    # force_meas = (force_meas - force_meas.mean(0)) / force_meas.std(0)

    # create training dataset
    obs_train = np.array([]).reshape(0, 2 * window_size)
    for i in range(walking_ranges[0][0], walking_ranges[0][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_train = np.concatenate(
            (obs_train, temp_obs), axis=0
        )

    ctrl_train = np.array([]).reshape(0, ctrl_horizon)
    out_train = np.array([]).reshape(0, (2+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[0][0], walking_ranges[0][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_train = np.concatenate((ctrl_train, temp_ctrl), axis=0)
        out_train = np.concatenate((out_train, temp_output), axis=0)

    # create validation dataset
    obs_val = np.array([]).reshape(0, 2 * window_size)
    for i in range(walking_ranges[1][0], walking_ranges[1][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_val = np.concatenate(
            (obs_val, temp_obs), axis=0
        )

    ctrl_val = np.array([]).reshape(0, ctrl_horizon)
    out_val = np.array([]).reshape(0, (2+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[1][0], walking_ranges[1][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_val = np.concatenate((ctrl_val, temp_ctrl), axis=0)
        out_val = np.concatenate((out_val, temp_output), axis=0)

    # create test dataset
    obs_test = np.array([]).reshape(0, 2 * window_size)
    for i in range(walking_ranges[2][0], walking_ranges[2][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_test = np.concatenate(
            (obs_test, temp_obs), axis=0
        )

    ctrl_test = np.array([]).reshape(0, ctrl_horizon)
    out_test = np.array([]).reshape(0, (2+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[2][0], walking_ranges[2][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_test = np.concatenate((ctrl_test, temp_ctrl), axis=0)
        out_test = np.concatenate((out_test, temp_output), axis=0)

    return (
        obs_train,
        ctrl_train,
        out_train,
        obs_val,
        ctrl_val,
        out_val,
        obs_test,
        ctrl_test,
        out_test,
    )


def generateDataWindowForce(window_size, ctrl_horizon):
    walking_ranges = np.array([[400, 2560], [3250, 5350], [7100, 9250]])
    w_d = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/data/walk_data_new.csv")
    Dankle = np.subtract(w_d["tibia_imu_rad_x"].to_numpy().flatten(), w_d["foot_imu_rad_x"].to_numpy().flatten())
    Dankle = Dankle.reshape((Dankle.shape[0], 1))
    observation = np.vstack((w_d["tibia_imu_rad_x"].to_numpy().flatten(), w_d["tibia_imu_gyr_x"].to_numpy().flatten())).T
    # observation = (observation - observation.mean(0)) / observation.std(0)
    print(f"observation mean of train data = {observation[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)}, std = {observation[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)}")
    print(f"observation mean of validation data = {observation[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)}, std = {observation[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)}")
    print(f"observation mean of test data = {observation[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)}, std = {observation[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)}")
    observation[
        walking_ranges[0][0]:walking_ranges[0][1]+10
        ] = (
            observation[walking_ranges[0][0]:walking_ranges[0][1]+10]
            - observation[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)
            )/observation[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)
    observation[
        walking_ranges[1][0]:walking_ranges[1][1]+10
        ] = (
            observation[walking_ranges[1][0]:walking_ranges[1][1]+10]
            - observation[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)
            )/observation[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)
    observation[
        walking_ranges[2][0]:walking_ranges[2][1]+10
        ] = (
            observation[walking_ranges[2][0]:walking_ranges[2][1]+10]
            - observation[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)
            )/observation[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)
    # observation = np.concatenate(
    #     (
    #         observation,
    #         Dankle,
    #     ),
    #     axis=1,
    # )
    # force_meas = w_d[["insole_sensor_1", "insole_sensor_ 2", "insole_sensor_3", "insole_sensor_4", "insole_sensor_5", "insole_sensor_6", "insole_sensor_7", "insole_sensor_8", "insole_sensor_9", "insole_sensor_10", "insole_sensor_11", "insole_sensor_12", "insole_sensor_13", "insole_sensor_14", "insole_sensor_15", "insole_sensor_16"]].to_numpy()
    force_meas = w_d[["insole_sensor_1", "insole_sensor_ 2", "insole_sensor_3", "insole_sensor_4"]].to_numpy()
    print(f"force_meas mean of train data = {force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)}, std = {force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)}")
    print(f"force_meas mean of validation data = {force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)}, std = {force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)}")
    print(f"force_meas mean of test data = {force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)}, std = {force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)}")
    force_meas[
        walking_ranges[0][0]:walking_ranges[0][1]+10
        ] = (
            force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10]
            - force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)
            )/force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)
    force_meas[
        walking_ranges[1][0]:walking_ranges[1][1]+10
        ] = (
            force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10]
            - force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)
            )/force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)
    force_meas[
        walking_ranges[2][0]:walking_ranges[2][1]+10
        ] = (
            force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10]
            - force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)
            )/force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)
    # force_meas = (force_meas - force_meas.mean(0)) / force_meas.std(0)

    # create training dataset
    obs_train = np.array([]).reshape(0, 2 * window_size)
    for i in range(walking_ranges[0][0], walking_ranges[0][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_train = np.concatenate(
            (obs_train, temp_obs), axis=0
        )

    ctrl_train = np.array([]).reshape(0, ctrl_horizon)
    out_train = np.array([]).reshape(0, (2+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[0][0], walking_ranges[0][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_train = np.concatenate((ctrl_train, temp_ctrl), axis=0)
        out_train = np.concatenate((out_train, temp_output), axis=0)

    # create validation dataset
    obs_val = np.array([]).reshape(0, 2 * window_size)
    for i in range(walking_ranges[1][0], walking_ranges[1][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_val = np.concatenate(
            (obs_val, temp_obs), axis=0
        )

    ctrl_val = np.array([]).reshape(0, ctrl_horizon)
    out_val = np.array([]).reshape(0, (2+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[1][0], walking_ranges[1][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_val = np.concatenate((ctrl_val, temp_ctrl), axis=0)
        out_val = np.concatenate((out_val, temp_output), axis=0)

    # create test dataset
    obs_test = np.array([]).reshape(0, 2 * window_size)
    for i in range(walking_ranges[2][0], walking_ranges[2][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_test = np.concatenate(
            (obs_test, temp_obs), axis=0
        )

    ctrl_test = np.array([]).reshape(0, ctrl_horizon)
    out_test = np.array([]).reshape(0, (2+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[2][0], walking_ranges[2][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_test = np.concatenate((ctrl_test, temp_ctrl), axis=0)
        out_test = np.concatenate((out_test, temp_output), axis=0)

    return (
        obs_train,
        np.concatenate((ctrl_train, out_train), axis=1),
        obs_val,
        np.concatenate((ctrl_val, out_val), axis=1),
        obs_test,
        np.concatenate((ctrl_test, out_test), axis=1),
    )


def build_network_block(regularizer_rate, layer_size, input, name):
    layer_list = [input]
    for i in range(len(layer_size)):
        activation = tf.nn.relu if i < len(layer_size) - 1 else None
        layer_list.append(
            layers.Dense(
                layer_size[i],
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(regularizer_rate),
                bias_regularizer=keras.regularizers.l2(regularizer_rate),
                name=f"{name}_layer_{i+1}",
            )(layer_list[i])
        )
    return layer_list[-1]


def buildModelWindow(
    data_size,
    ctrl_layer_size,
    pred_layer_size,
    regularizer_rate=0.001,
):

    input_layer = tf.keras.Input(shape=(data_size[1]))
    last_ctrl_layer = build_network_block(
        regularizer_rate, ctrl_layer_size, input_layer, "ctrl"
    )
    # concat_inp = tf.gather(input_layer, indices=[36, 37, 38, 39])
    last_pred_layer = build_network_block(
        regularizer_rate,
        pred_layer_size,
        layers.Concatenate()([last_ctrl_layer, input_layer[:, -4:]]),  # we only input the state of last step instead of the last 10 steps
        # layers.Concatenate()([last_ctrl_layer, input_layer]),
        "pred",
    )

    model = Model(
        inputs=[input_layer],
        outputs=[last_pred_layer, last_ctrl_layer],
        name="seq_control_predictor_NN",
    )

    def loss2(y_true, y_pred):
        loss = tf.keras.losses.MSE(y_true, y_pred)
        return loss

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[loss2, loss2],
        # metrics=["accuracy"],
    )

    model.summary()
    architecture = model.to_json()
    filepath = "models/model1"
    tf_callback = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weight_only=False,
            mode="auto",
            save_freq="epoch",
            options=None,
        ),
        keras.callbacks.TensorBoard(log_dir="tf_logs"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=20, min_lr=0.0001,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True
        ),
    ]

    return model, tf_callback, architecture


def plotTestData(model, test_obs, test_ctrls, control_horizon):
    pred_ctrls = model.predict(test_obs)

    # subplots
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(test_ctrls[:, 0].flatten(), label="test control", color="#173f5f")
    # axs[0].plot(pred_ctrls[1].flatten(), label="prediction control", color=[0.4705, 0.7921, 0.6470])
    # create a list of different colors to the size of control_horizon
    colors = plt.cm.jet(np.linspace(0, 1, 1))
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[:, 1])
    x_lim = 271
    for i in range(1):
        ax5.plot(
            test_ctrls[:, i].flatten()[:x_lim],
            # label=f"test control {i+1}",
            color=colors[i],
            linestyle="dashed",
        )
        ax5.plot(
            pred_ctrls[1][:, i].flatten()[:x_lim],
            label=f"Control {i+1}",
            color=colors[i],
        )
    # ax5.set_title("Control")
    for i in range(1):
        ax1.plot(
            test_ctrls[:, 4*i+control_horizon].flatten()[:x_lim],
            # label="test femur ang.",
            color=colors[i],
            linestyle="dashed",
        )
        ax1.plot(
            pred_ctrls[0][:, 4*i].flatten()[:x_lim],
            label=f"Femur ang. {i+1}",
            color=colors[i],
        )
        ax2.plot(
            test_ctrls[:, 4*i+1+control_horizon].flatten()[:x_lim],
            # label="test femur ang. vel.",
            color=colors[i],
            linestyle="dashed",
        )
        ax2.plot(
            pred_ctrls[0][:, 4*i+1].flatten()[:x_lim],
            label=f"Femur ang. vel. {i+1}",
            color=colors[i],
        )
        ax3.plot(
            test_ctrls[:, 4*i+2+control_horizon].flatten()[:x_lim],
            # label="test tibia ang.",
            color=colors[i],
            linestyle="dashed",
        )
        ax3.plot(
            pred_ctrls[0][:, 4*i+2].flatten()[:x_lim],
            label=f"Tibia ang. {i+1}",
            color=colors[i],
        )
        ax4.plot(
            test_ctrls[:, 4*i+3+control_horizon].flatten()[:x_lim],
            # label="test tibia ang. vel.",
            color=colors[i],
            linestyle="dashed",
        )
        ax4.plot(
            pred_ctrls[0][:, 4*i+3].flatten()[:x_lim],
            label=f"Tibia ang. vel. {i+1}",
            color=colors[i],
        )
    ax5.set_xlabel("time step")
    ax5.set_ylabel("ankle angle [deg]")
    ax5.set_xlim(0, x_lim)
    ax1.set_ylabel("Femur ang. \n[deg]")
    ax1.set_xlim(0, x_lim)
    # ax1.set_title("Femur Angle")
    ax2.set_ylabel("Femur ang. vel. \n [deg/s]")
    ax2.set_xlim(0, x_lim)
    # ax2.set_title("Femur Angular Velocity")
    ax3.set_ylabel("Tibia ang. \n [deg]")
    ax3.set_xlim(0, x_lim)
    # ax3.set_title("Tibia Angle")
    ax4.set_xlabel("time step")
    ax4.set_ylabel("Tibia ang. vel. \n [deg/s]")
    ax4.set_xlim(0, x_lim)
    # ax4.set_title("Tibia Angular Velocity")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    # lines, labels = ax5.get_legend_handles_labels()
    # leg = fig.legend(
    #     lines,
    #     labels,
    #     loc="center",
    #     bbox_to_anchor=(0.5, -0.5),
    #     # bbox_to_anchor=(0.75, 0.65),
    #     bbox_transform=fig.transFigure,
    #     ncol=1,
    #     fontsize=14,
    # )
    # leg.get_frame().set_facecolor("white")
    # plt.tight_layout()
    plt.show()


def plotTestDataForce(model, test_obs, test_ctrls, control_horizon):
    pred_ctrls = model.predict(test_obs)

    # subplots
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(test_ctrls[:, 0].flatten(), label="test control", color="#173f5f")
    # axs[0].plot(pred_ctrls[1].flatten(), label="prediction control", color=[0.4705, 0.7921, 0.6470])
    # create a list of different colors to the size of control_horizon
    colors = plt.cm.jet(np.linspace(0, 1, 18))
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(5, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[4, 0])
    ax6 = fig.add_subplot(gs[:, 1])
    x_lim = test_obs.shape[0]
    ax6.plot(
        test_ctrls[:, 0].flatten()[:x_lim],
        # label=f"test control {i+1}",
        color=colors[0],
        linestyle="dashed",
    )
    ax6.plot(
        pred_ctrls[1][:, 0].flatten()[:x_lim],
        label=f"Control {1}",
        color=colors[0],
    )
    # ax5.set_title("Control")
    for i in range(0, 2):
        ax1.plot(
            test_ctrls[:, i+control_horizon].flatten()[:x_lim],
            # label="test femur ang.",
            color=colors[i],
            linestyle="dashed",
        )
        ax1.plot(
            pred_ctrls[0][:, i].flatten()[:x_lim],
            label=f"State {i+1}",
            color=colors[i],
        )

    for i in range(2, 6):
        ax2.plot(
            test_ctrls[:, i+control_horizon].flatten()[:x_lim],
            # label="test femur ang. vel.",
            color=colors[i],
            linestyle="dashed",
        )
        ax2.plot(
            pred_ctrls[0][:, i].flatten()[:x_lim],
            label=f"State {i+1}",
            color=colors[i],
        )
    
    for i in range(6, 10):
        ax3.plot(
            test_ctrls[:, i+control_horizon].flatten()[:x_lim],
            # label="test femur ang. vel.",
            color=colors[i],
            linestyle="dashed",
        )
        ax3.plot(
            pred_ctrls[0][:, i].flatten()[:x_lim],
            label=f"State {i+1}",
            color=colors[i],
        )
    
    for i in range(10, 14):
        ax4.plot(
            test_ctrls[:, i+control_horizon].flatten()[:x_lim],
            # label="test femur ang. vel.",
            color=colors[i],
            linestyle="dashed",
        )
        ax4.plot(
            pred_ctrls[0][:, i].flatten()[:x_lim],
            label=f"State {i+1}",
            color=colors[i],
        )
    
    for i in range(14, 18):
        ax5.plot(
            test_ctrls[:, i+control_horizon].flatten()[:x_lim],
            # label="test femur ang. vel.",
            color=colors[i],
            linestyle="dashed",
        )
        ax5.plot(
            pred_ctrls[0][:, i].flatten()[:x_lim],
            label=f"State {i+1}",
            color=colors[i],
        )

    ax5.set_xlabel("time step")
    # ax5.set_ylabel("ankle angle [deg]")
    ax5.set_xlim(0, x_lim)
    ax1.set_ylabel("Femur ang. \n[deg]")
    ax1.set_xlim(0, x_lim)
    # ax1.set_title("Femur Angle")
    ax2.set_ylabel("Femur ang. vel. \n [deg/s]")
    ax2.set_xlim(0, x_lim)
    # ax2.set_title("Femur Angular Velocity")
    ax3.set_ylabel("Tibia ang. \n [deg]")
    ax3.set_xlim(0, x_lim)
    # ax3.set_title("Tibia Angle")
    ax4.set_xlabel("time step")
    ax4.set_ylabel("Tibia ang. vel. \n [deg/s]")
    ax4.set_xlim(0, x_lim)
    # ax4.set_title("Tibia Angular Velocity")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    # lines, labels = ax5.get_legend_handles_labels()
    # leg = fig.legend(
    #     lines,
    #     labels,
    #     loc="center",
    #     bbox_to_anchor=(0.5, -0.5),
    #     # bbox_to_anchor=(0.75, 0.65),
    #     bbox_transform=fig.transFigure,
    #     ncol=1,
    #     fontsize=14,
    # )
    # leg.get_frame().set_facecolor("white")
    # plt.tight_layout()
    plt.show()


def plotTestDataSubForce(model, test_obs, test_ctrls, control_horizon):
    pred_ctrls = model.predict(test_obs)

    # subplots
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(test_ctrls[:, 0].flatten(), label="test control", color="#173f5f")
    # axs[0].plot(pred_ctrls[1].flatten(), label="prediction control", color=[0.4705, 0.7921, 0.6470])
    # create a list of different colors to the size of control_horizon
    colors = plt.cm.jet(np.linspace(0, 1, 18))
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    x_lim = test_obs.shape[0]
    ax3.plot(
        test_ctrls[:, 0].flatten()[:x_lim],
        # label=f"test control {i+1}",
        color=colors[0],
        linestyle="dashed",
    )
    ax3.plot(
        pred_ctrls[1][:, 0].flatten()[:x_lim],
        label=f"Control {1}",
        color=colors[0],
    )
    # ax5.set_title("Control")
    for i in range(0, 4):
        ax1.plot(
            test_ctrls[:, i+control_horizon].flatten()[:x_lim],
            # label="test femur ang.",
            color=colors[i],
            linestyle="dashed",
        )
        ax1.plot(
            pred_ctrls[0][:, i].flatten()[:x_lim],
            label=f"State {i+1}",
            color=colors[i],
        )

    for i in range(4, 8):
        ax2.plot(
            test_ctrls[:, i+control_horizon].flatten()[:x_lim],
            # label="test femur ang. vel.",
            color=colors[i],
            linestyle="dashed",
        )
        ax2.plot(
            pred_ctrls[0][:, i].flatten()[:x_lim],
            label=f"State {i+1}",
            color=colors[i],
        )

    

    ax1.set_ylabel("Femur ang. \n[deg]")
    ax1.set_xlim(0, x_lim)
    # ax1.set_title("Femur Angle")
    ax2.set_ylabel("Femur ang. vel. \n [deg/s]")
    ax2.set_xlim(0, x_lim)
    # ax2.set_title("Femur Angular Velocity")
    ax3.set_ylabel("Tibia ang. \n [deg]")
    ax3.set_xlim(0, x_lim)
    # ax4.set_title("Tibia Angular Velocity")
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # lines, labels = ax5.get_legend_handles_labels()
    # leg = fig.legend(
    #     lines,
    #     labels,
    #     loc="center",
    #     bbox_to_anchor=(0.5, -0.5),
    #     # bbox_to_anchor=(0.75, 0.65),
    #     bbox_transform=fig.transFigure,
    #     ncol=1,
    #     fontsize=14,
    # )
    # leg.get_frame().set_facecolor("white")
    # plt.tight_layout()
    plt.show()


def hand_labeling_data(train_out, test_out, control_horizon, state_bounds, gap=0.2, num_states=6):
    for key, value in state_bounds.items():
        for i in range(control_horizon):
            train_out[:, control_horizon + num_states*i+key] = np.clip(
                train_out[:, control_horizon + num_states*i+key],
                value[0]+gap, value[1]-gap
                )
            test_out[:, control_horizon + num_states*i+key] = np.clip(
                test_out[:, control_horizon + num_states*i+key],
                value[0]+gap, value[1]-gap
                )
    return train_out, test_out


def generateDataWindowForceNonObs(window_size, ctrl_horizon):
    pad = window_size + ctrl_horizon
    # walking_ranges = np.array([[600, 2760-pad], [3500, 6045-pad], [7000, 9500-pad]])
    walking_ranges = np.array([[7185, 9000-pad], [9995, 11150-pad], [11500, 12530-pad]])
    # walking_ranges = np.array([[33270, 34955-pad], [37765, 39340-pad], [40270, 42000-pad]])
    w_d = loadData("/Users/keyvanmajd/state_control_repair/demos/examples/tc7_walking_example/repair_bound_constraint/data/data_full.csv") 
    Dankle = w_d[:,37] # ankle angle
    Dankle = Dankle.reshape((Dankle.shape[0], 1))
    observation = w_d[:,[14, 18, 7, 11]] # femur angle, vel, tibia angle, vel
    # observation = (observation - observation.mean(0)) / observation.std(0)
    print(f"observation mean of train data = {observation[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)}, std = {observation[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)}")
    print(f"observation mean of validation data = {observation[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)}, std = {observation[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)}")
    print(f"observation mean of test data = {observation[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)}, std = {observation[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)}")
    observation[
        walking_ranges[0][0]:walking_ranges[0][1]+pad
        ] = (
            observation[walking_ranges[0][0]:walking_ranges[0][1]+pad]
            - observation[walking_ranges[0][0]:walking_ranges[0][1]+pad].mean(0)
            )/observation[walking_ranges[0][0]:walking_ranges[0][1]+pad].std(0)
    observation[
        walking_ranges[1][0]:walking_ranges[1][1]+pad
        ] = (
            observation[walking_ranges[1][0]:walking_ranges[1][1]+pad]
            - observation[walking_ranges[1][0]:walking_ranges[1][1]+pad].mean(0)
            )/observation[walking_ranges[1][0]:walking_ranges[1][1]+pad].std(0)
    observation[
        walking_ranges[2][0]:walking_ranges[2][1]+pad
        ] = (
            observation[walking_ranges[2][0]:walking_ranges[2][1]+pad]
            - observation[walking_ranges[2][0]:walking_ranges[2][1]+pad].mean(0)
            )/observation[walking_ranges[2][0]:walking_ranges[2][1]+pad].std(0)
#     observation = np.concatenate(
#         (
#             observation,
#             Dankle,
#         ),
#         axis=1,
#     )
    # force_meas = w_d[["insole_sensor_1", "insole_sensor_ 2", "insole_sensor_3", "insole_sensor_4", "insole_sensor_5", "insole_sensor_6", "insole_sensor_7", "insole_sensor_8", "insole_sensor_9", "insole_sensor_10", "insole_sensor_11", "insole_sensor_12", "insole_sensor_13", "insole_sensor_14", "insole_sensor_15", "insole_sensor_16"]].to_numpy()
    force_meas = w_d[:, [21, 22, 23, 24]] # insole sensors 1, 2, 3, and 4
    print(f"force_meas mean of train data = {force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].mean(0)}, std = {force_meas[walking_ranges[0][0]:walking_ranges[0][1]+10].std(0)}")
    print(f"force_meas mean of validation data = {force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].mean(0)}, std = {force_meas[walking_ranges[1][0]:walking_ranges[1][1]+10].std(0)}")
    print(f"force_meas mean of test data = {force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].mean(0)}, std = {force_meas[walking_ranges[2][0]:walking_ranges[2][1]+10].std(0)}")
    force_meas[
        walking_ranges[0][0]:walking_ranges[0][1]+pad
        ] = (
            force_meas[walking_ranges[0][0]:walking_ranges[0][1]+pad]
            - force_meas[walking_ranges[0][0]:walking_ranges[0][1]+pad].mean(0)
            )/force_meas[walking_ranges[0][0]:walking_ranges[0][1]+pad].std(0)
    force_meas[
        walking_ranges[1][0]:walking_ranges[1][1]+pad
        ] = (
            force_meas[walking_ranges[1][0]:walking_ranges[1][1]+pad]
            - force_meas[walking_ranges[1][0]:walking_ranges[1][1]+pad].mean(0)
            )/force_meas[walking_ranges[1][0]:walking_ranges[1][1]+pad].std(0)
    force_meas[
        walking_ranges[2][0]:walking_ranges[2][1]+pad
        ] = (
            force_meas[walking_ranges[2][0]:walking_ranges[2][1]+pad]
            - force_meas[walking_ranges[2][0]:walking_ranges[2][1]+pad].mean(0)
            )/force_meas[walking_ranges[2][0]:walking_ranges[2][1]+pad].std(0)
    # force_meas = (force_meas - force_meas.mean(0)) / force_meas.std(0)

    # create training dataset
    obs_train = np.array([]).reshape(0, 4 * window_size)
    for i in range(walking_ranges[0][0], walking_ranges[0][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_train = np.concatenate(
            (obs_train, temp_obs), axis=0
        )

    ctrl_train = np.array([]).reshape(0, ctrl_horizon)
    out_train = np.array([]).reshape(0, (4 + force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[0][0], walking_ranges[0][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_train = np.concatenate((ctrl_train, temp_ctrl), axis=0)
        out_train = np.concatenate((out_train, temp_output), axis=0)

    # create validation dataset
    obs_val = np.array([]).reshape(0, 4 * window_size)
    for i in range(walking_ranges[1][0], walking_ranges[1][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_val = np.concatenate(
            (obs_val, temp_obs), axis=0
        )

    ctrl_val = np.array([]).reshape(0, ctrl_horizon)
    out_val = np.array([]).reshape(0, (4+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[1][0], walking_ranges[1][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_val = np.concatenate((ctrl_val, temp_ctrl), axis=0)
        out_val = np.concatenate((out_val, temp_output), axis=0)

    # create test dataset
    obs_test = np.array([]).reshape(0, 4 * window_size)
    for i in range(walking_ranges[2][0], walking_ranges[2][1]):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observation[i + j, :].reshape(1, -1)), axis=1
            )
        obs_test = np.concatenate(
            (obs_test, temp_obs), axis=0
        )

    ctrl_test = np.array([]).reshape(0, ctrl_horizon)
    out_test = np.array([]).reshape(0, (4+force_meas.shape[1]) * ctrl_horizon)
    for i in range(walking_ranges[2][0], walking_ranges[2][1]):
        temp_ctrl = np.array([]).reshape(1, 0)
        temp_output = np.array([]).reshape(1, 0)
        for j in range(ctrl_horizon):
            temp_ctrl = np.concatenate(
                (temp_ctrl, Dankle[i + window_size + j].reshape(1, -1)),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    observation[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
            temp_output = np.concatenate(
                (
                    temp_output,
                    force_meas[i + window_size + j, :].reshape(1, -1),
                ),
                axis=1,
            )
        ctrl_test = np.concatenate((ctrl_test, temp_ctrl), axis=0)
        out_test = np.concatenate((out_test, temp_output), axis=0)

    return (
        obs_train,
        ctrl_train,
        out_train,
        obs_val,
        ctrl_val,
        out_val,
        obs_test,
        ctrl_test,
        out_test,
    )


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=5120
                    )
                ],
            )
        except RuntimeError as e:
            print(e)
    now = datetime.now()
    now_str = f"_{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}_{now.second}"
    num_states_obs = 4        # tibia angle and angular velocity
    num_states_non_obs = 4   # 16 force sensors
    num_hidden_nodes = 256
    batch_size = 5
    epochs = 500
    control_horizon = 10
    ctrl_layer_size = [num_hidden_nodes, num_hidden_nodes, control_horizon]
    pred_layer_size = [
        num_hidden_nodes,
        # num_hidden_nodes,
        (num_states_obs + num_states_non_obs) * control_horizon
        ]
    orig_load = False
    trained_time_str = "_2_25_2023_1_42_2"
    # Train window model
    # (
    #     train_obs,
    #     train_out,
    #     test_obs,
    #     test_out,
    # ) = generateDataWindow(10, control_horizon)
    # (
    #     train_obs,
    #     train_out,
    #     val_obs,
    #     val_out,
    #     test_obs,
    #     test_out,
    # ) = generateDataWindowForce(10, control_horizon)

    (
        s_h_train,
        a_p_train,
        s_p_train,
        s_h_val,
        a_p_val,
        s_p_val,
        s_h_test,
        a_p_test,
        s_p_test,
    ) = generateDataWindowForce2(10, control_horizon)
    train_obs = s_h_train
    train_out = np.concatenate((a_p_train, s_p_train), axis=1)
    val_obs = s_h_val
    val_out = np.concatenate((a_p_val, s_p_val), axis=1)
    test_obs = s_h_test
    test_out = np.concatenate((a_p_test, s_p_test), axis=1)

    if orig_load:
        ctrl_model_orig = keras.models.load_model(
            os.path.dirname(os.path.realpath(__file__))
            + f"/models/model_ctrl_pred_{control_horizon}_{num_hidden_nodes}{trained_time_str}"
        )
    else:
        train_obs = np.vstack((train_obs, val_obs))
        train_out = np.vstack((train_out, val_out))
        train_obs = np.vstack((train_obs, test_obs))
        train_out = np.vstack((train_out, test_out))
        _, callback, architecture = buildModelWindow(
            train_obs.shape, ctrl_layer_size, pred_layer_size, 0.001
        )
        ctrl_model_orig = keras.models.load_model(
            os.path.dirname(os.path.realpath(__file__))
            + f"/models/model_ctrl_pred_{control_horizon}_{num_hidden_nodes}{trained_time_str}"
        )

        def loss2(y_true, y_pred):
            loss = tf.keras.losses.MSE(y_true, y_pred)
            return loss

        ctrl_model_orig.compile(
            optimizer=keras.optimizers.Adam(),
            loss=[loss2, loss2],
            # metrics=["accuracy"],
        )
        ctrl_model_orig.fit(
            train_obs,
            [train_out[:, control_horizon:], train_out[:, 0:control_horizon]],
            validation_data=(
                test_obs, [test_out[:, control_horizon:], test_out[:, 0:control_horizon]]
                ),
            batch_size=batch_size,
            epochs=epochs,
            use_multiprocessing=True,
            verbose=1,
            shuffle=False,
            callbacks=callback,
        )
        model_name = f"model_ctrl_pred_{control_horizon}_{num_hidden_nodes}{now_str}"
        print(model_name)
        keras.models.save_model(
            ctrl_model_orig,
            os.path.dirname(os.path.realpath(__file__))
            + f"/models/{model_name}",
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None,
            save_traces=True,
        )
    plotTestDataSubForce(ctrl_model_orig, train_obs, train_out, control_horizon)
    # plotTestDataSubForce(ctrl_model_orig, val_obs, val_out, control_horizon)
    plotTestDataSubForce(ctrl_model_orig, test_obs, test_out, control_horizon)

    # finetune model with new parameters
    print(f"weight norm layer {1} is {np.linalg.norm(ctrl_model_orig.layers[1].get_weights()[0])}")
    print(f"weight norm layer {2} is {np.linalg.norm(ctrl_model_orig.layers[2].get_weights()[0])}")
    print(f"weight norm layer {3} is {np.linalg.norm(ctrl_model_orig.layers[3].get_weights()[0])}")
    print(f"weight norm layer {6} is {np.linalg.norm(ctrl_model_orig.layers[6].get_weights()[0])}")
    print(f"weight norm layer {7} is {np.linalg.norm(ctrl_model_orig.layers[7].get_weights()[0])}")
    ctrl_model_orig.layers[1].trainable = True
    ctrl_model_orig.layers[2].trainable = True
    ctrl_model_orig.layers[3].trainable = True
    ctrl_model_orig.layers[5].trainable = False
    ctrl_model_orig.layers[6].trainable = False

    def loss2(y_true, y_pred):
        loss = tf.keras.losses.MSE(y_true, y_pred)
        return loss

    def loss3(y_true, y_pred):
        loss = 0
        const_state = [4, 5, 6, 7]
        for i in range((num_states_obs+num_states_non_obs)*control_horizon):
            # if i % (num_states_obs+num_states_non_obs) == const_state:
            if i % (num_states_obs+num_states_non_obs) in const_state:
                loss += 10*tf.keras.losses.MSE(y_true[:, i], y_pred[:, i])
            else:
                loss += 1*tf.keras.losses.MSE(y_true[:, i], y_pred[:, i])
        return loss

    ctrl_model_orig.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=[loss3, loss2],
        # metrics=["accuracy"],
    )
    # handlabeling the data
    train_out3, val_out3 = hand_labeling_data(
        train_out,
        val_out,
        control_horizon,
        {4: [-1.4, 1], 5: [-1.4, 1], 6: [-1.4, 1], 7: [-1.4, 1]},
        0.3,
        num_states_obs + num_states_non_obs,
        )
    callback = [
        keras.callbacks.TensorBoard(log_dir="tf_logs"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=15, min_lr=0.0001,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=400, restore_best_weights=True
        ),
    ]
    ctrl_model_orig.fit(
        train_obs3,
        [train_out3[:, control_horizon:], train_out3[:, 0:control_horizon]],
        validation_data=(
            val_obs3, [val_out3[:, control_horizon:], val_out3[:, 0:control_horizon]]
            ),
        batch_size=batch_size,
        epochs=500,
        use_multiprocessing=True,
        verbose=1,
        shuffle=False,
        callbacks=callback,
    )
    print(f"weight norm layer {1} is {np.linalg.norm(ctrl_model_orig.layers[1].get_weights()[0])}")
    print(f"weight norm layer {2} is {np.linalg.norm(ctrl_model_orig.layers[2].get_weights()[0])}")
    print(f"weight norm layer {3} is {np.linalg.norm(ctrl_model_orig.layers[3].get_weights()[0])}")
    print(f"weight norm layer {6} is {np.linalg.norm(ctrl_model_orig.layers[6].get_weights()[0])}")
    print(f"weight norm layer {7} is {np.linalg.norm(ctrl_model_orig.layers[7].get_weights()[0])}")
    model_name = f"model_ctrl_pred_{control_horizon}_{num_hidden_nodes}{now_str}"
    keras.models.save_model(
        ctrl_model_orig,
        os.path.dirname(os.path.realpath(__file__))
        + f"/models/{model_name}",
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )
    print("saved: model")
    plotTestDataSubForce(ctrl_model_orig, train_obs3, train_out3, control_horizon)
    print(f'model name: {model_name}')
