import os
from tensorflow import keras
import matplotlib.pyplot as plt
from sarp.utils import (
    load_expert_data_prosthesis
)


if __name__ == "__main__":
    ctrl_horizon = 30
    window_size = 10
    batch_size = 32
    num_epochs = 100
    num_hidden = 1028
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

    # build model
    model = keras.Sequential(
        [
            keras.layers.Dense(num_hidden, activation="relu"),
            keras.layers.Dense(num_hidden, activation="relu"),
            keras.layers.Dense(a_train.shape[1], activation="linear"),
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
            monitor="val_loss", patience=30, restore_best_weights=True
        ),
    ]

    model.fit(
        s_train,
        a_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(s_val, a_val),
        callbacks=tf_callback,
    )
    a_pred_train = model.predict(s_train)
    a_pred_test = model.predict(s_test)
    a_pred_val = model.predict(s_val)

    # plot training results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(a_train[:, 0], label="expert")
    plt.plot(a_pred_train[:, 0], label="model")
    plt.legend()
    plt.title("Training")
    plt.subplot(1, 3, 2)
    plt.plot(a_test[:, 0], label="expert")
    plt.plot(a_pred_test[:, 0], label="model")
    plt.legend()
    plt.title("Testing")
    plt.subplot(1, 3, 3)
    plt.plot(a_val[:, 0], label="expert")
    plt.plot(a_pred_val[:, 0], label="model")
    plt.legend()
    plt.title("Validation")
    plt.show()


    # save model
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(current_dir + f"/trained_models/policy_partial_obs"):
        os.makedirs(current_dir + f"/trained_models/policy_partial_obs")
    keras.models.save_model(
        model,
        f"{current_dir}/trained_models/policy_partial_obs/model",
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )