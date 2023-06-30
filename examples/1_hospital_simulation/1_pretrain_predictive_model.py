import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sarp.utils import load_wander_data, separate_train_test, load_expert_data

if __name__ == "__main__":
    ############################
    ## train transition model
    batch_size_tran = 32
    num_epochs_tran = 100
    num_hidden_tran = 256
    learning_rate_tran = 0.001
    train_ratio_tran = 0.8

    # load the expert data
    data_dir = os.path.dirname(os.path.realpath(__file__)) + f"/data/expert_data"
    num_samples = len(os.listdir(data_dir))

    state, _, next_state, _ = load_expert_data(data_dir, num_samples)
    train_data, test_data = separate_train_test(
        [state, next_state], test_ratio=1 - train_ratio_tran
    )

    state_train, next_state_train = train_data
    state_test, next_state_test = test_data

    # build model
    model_tran = keras.Sequential(
        [
            keras.layers.Dense(num_hidden_tran, activation="relu"),
            keras.layers.Dense(num_hidden_tran, activation="relu"),
            keras.layers.Dense(next_state_train[0].shape[-1], activation="linear"),
        ]
    )
    model_tran.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate_tran),
        loss="mse",
        metrics=["mse"],
    )

    # train model
    tf_callback = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            min_lr=0.00001,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]
    model_tran.fit(
        tf.concat(state_train, 0),
        tf.concat(next_state_train, 0),
        batch_size=batch_size_tran,
        epochs=num_epochs_tran,
        validation_data=(tf.concat(state_test, 0), tf.concat(next_state_test, 0)),
        callbacks=tf_callback,
    )

    ############################
    ## train collision model
    batch_size_col = 32
    num_epochs_col = 100
    num_hidden_col = 128
    learning_rate_col = 0.001
    train_ratio_col = 0.8

    # load the collision data
    data_dir = os.path.dirname(os.path.realpath(__file__)) + f"/data/wander_data"
    num_samples = len(os.listdir(data_dir))
    hits, scans = load_wander_data(data_dir, num_samples)

    # shuffle hits and scans lists
    c = list(zip(hits, scans))
    np.random.shuffle(c)
    hits, scans = zip(*c)
    train_scans = np.vstack(scans[: int(train_ratio_col * len(scans))])
    train_collision = np.vstack(hits[: int(train_ratio_col * len(hits))])
    test_scans = np.vstack(scans[int(train_ratio_col * len(scans)) :])
    test_collision = np.vstack(hits[int(train_ratio_col * len(hits)) :])

    # load network
    model_col = keras.Sequential(
        [
            keras.layers.Dense(num_hidden_col, activation="relu"),
            keras.layers.Dense(num_hidden_col, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model_col.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # callbacks
    tf_callback = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.1, patience=5, min_lr=0.00001
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        ),
    ]

    # train collision model
    model_col.fit(
        train_scans,
        train_collision,
        validation_data=(test_scans, test_collision),
        epochs=num_epochs_col,
        batch_size=batch_size_col,
        callbacks=tf_callback,
    )

    ############################
    # merge the model_tran and model_col in series
    model_predictive = keras.Sequential(
        [
            keras.layers.Dense(num_hidden_tran, activation="relu"),
            keras.layers.Dense(num_hidden_tran, activation="relu"),
            keras.layers.Dense(next_state_train[0].shape[-1], activation="linear"),
            keras.layers.Dense(num_hidden_col, activation="relu"),
            keras.layers.Dense(num_hidden_col, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model_predictive.predict(state_test[0][0:1])
    # substitute the weights of model_tran and model_col
    model_predictive.layers[0].set_weights(model_tran.layers[0].get_weights())
    model_predictive.layers[1].set_weights(model_tran.layers[1].get_weights())
    model_predictive.layers[2].set_weights(model_tran.layers[2].get_weights())
    model_predictive.layers[3].set_weights(model_col.layers[0].get_weights())
    model_predictive.layers[4].set_weights(model_col.layers[1].get_weights())
    model_predictive.layers[5].set_weights(model_col.layers[2].get_weights())
    # save the model
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(current_dir + f"/trained_models/predictive_model"):
        os.makedirs(current_dir + f"/trained_models/predictive_model")
    keras.models.save_model(
        model_predictive,
        f"{current_dir}/trained_models/predictive_model/model",
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )
