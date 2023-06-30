import os
from tensorflow import keras
import tensorflow as tf
from sarp.utils import load_expert_data, separate_train_test, curriculum_training

if __name__ == "__main__":
    batch_size = 32
    num_epochs = 100
    num_hidden = 256
    learning_rate = 0.001

    # load the expert data
    data_dir = os.path.dirname(os.path.realpath(__file__)) + f"/data/expert_data"
    num_samples = len(os.listdir(data_dir))

    state, action, _, _ = load_expert_data(data_dir, num_samples)
    train_data, test_data = separate_train_test([state, action], test_ratio=0.2)

    state_train, action_train = train_data
    state_test, action_test = test_data

    # build model
    model = keras.Sequential(
        [
            keras.layers.Dense(num_hidden, activation="relu"),
            keras.layers.Dense(num_hidden, activation="relu"),
            keras.layers.Dense(action_train[0].shape[-1], activation="linear"),
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
            patience=5,
            min_lr=0.00001,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]
    # model.fit(
    #     tf.concat(state_train, 0),
    #     tf.concat(action_train, 0),
    #     batch_size=batch_size,
    #     epochs=num_epochs,
    #     validation_data=(tf.concat(state_test, 0), tf.concat(action_test, 0)),
    #     callbacks=tf_callback,
    # )
    curriculum_training(
        model,
        state_train,
        action_train,
        state_test,
        action_test,
        epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    # save model
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(current_dir + f"/trained_models/policy"):
        os.makedirs(current_dir + f"/trained_models/policy")
    keras.models.save_model(
        model,
        f"{current_dir}/trained_models/policy/model",
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )
