import os
from tensorflow import keras
import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sarp.utils import (
    load_expert_data_prosthesis
)


if __name__ == "__main__":
    ctrl_horizon = 30
    window_size = 10
    batch_size = 32
    num_epochs = 100
    num_hidden = 512
    learning_rate = 0.001

    # data directory
    data_dir = os.path.dirname(os.path.realpath(__file__)) + f"/data/expert_data/sample1"

    data = load_expert_data_prosthesis(data_dir, 10, 30, type="partial_obs")
    (
        s_train,
        a_train,
        p_train,
        s_val,
        a_val,
        p_val,
        s_test,
        a_test,
        p_test,
    ) = data
    model_inp_train = np.concatenate((s_train[:,-4:], a_train), axis=1)
    model_inp_val = np.concatenate((s_val[:,-4:], a_val), axis=1)
    model_inp_test = np.concatenate((s_test[:,-4:], a_test), axis=1)
    # model_inp_train = a_train
    # model_inp_val = a_val
    # model_inp_test = a_test

    # build model
    model = keras.Sequential(
        [
            keras.layers.Dense(num_hidden, activation="relu"),
            keras.layers.Dense(num_hidden, activation="relu"),
            keras.layers.Dense(p_train.shape[1], activation="linear"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mse"],
    )

    # train model
    tf_callback = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=20,
            min_lr=0.00001,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=False
        ),
    ]

    model.fit(
        model_inp_train,
        p_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(model_inp_test, p_test),
        callbacks=tf_callback,
    )
    p_pred_train = model.predict(model_inp_train)
    p_pred_test = model.predict(model_inp_test)
    p_pred_val = model.predict(model_inp_val)

    # plot training results and specify xlim
    x_lim_max = 500
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 3, 1)
    plt.plot(p_train[:, 0], label="expert")
    plt.plot(p_pred_train[:, 0], label="model")
    plt.xlim([0, x_lim_max])
    plt.legend()
    plt.title("Training")
    plt.subplot(2, 3, 2)
    plt.plot(p_test[:, 0], label="expert")
    plt.plot(p_pred_test[:, 0], label="model")
    plt.xlim([0, x_lim_max])
    plt.legend()
    plt.title("Testing")
    plt.subplot(2, 3, 3)
    plt.plot(p_val[:, 0], label="expert")
    plt.plot(p_pred_val[:, 0], label="model")
    plt.xlim([0, x_lim_max])
    plt.legend()
    plt.title("Validation")
    plt.subplot(2, 3, 4)
    plt.plot(a_train[:, 0], label="expert")
    plt.xlim([0, x_lim_max])
    plt.legend()
    plt.subplot(2, 3, 5)
    plt.plot(a_test[:, 0], label="expert")
    plt.xlim([0, x_lim_max])
    plt.legend()
    plt.subplot(2, 3, 6)
    plt.plot(a_val[:, 0], label="expert")
    plt.legend()
    plt.xlim([0, x_lim_max])
    plt.show()


    # save model
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(current_dir + f"/trained_models/predictive_partial_obs"):
        os.makedirs(current_dir + f"/trained_models/predictive_partial_obs")
    keras.models.save_model(
        model,
        f"{current_dir}/trained_models/predictive_partial_obs/model",
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )
