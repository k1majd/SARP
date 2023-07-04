import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class state_col_block(tf.keras.Model):
    def __init__(self, regularizer_rate, ctrl_arch, trans_arch, col_arch):
        super(state_col_block, self).__init__(name="")
        self.layer_list = []
        self.ctrl_arch = ctrl_arch
        self.col_arch = col_arch
        self.trans_arch = trans_arch
        # construct controller network
        self.layer_list.append(
            tf.keras.layers.Dense(
                ctrl_arch[1],
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                input_shape=(ctrl_arch[0],),
                name="policy_layer_1",
            )
        )
        for i in range(2, len(ctrl_arch) - 1):
            self.layer_list.append(
                tf.keras.layers.Dense(
                    ctrl_arch[i],
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    name=f"policy_layer_{i}",
                )
            )
        self.layer_list.append(
            tf.keras.layers.Dense(
                ctrl_arch[-1],
                activation="linear",
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                name=f"policy_layer_{len(ctrl_arch)-1}",
            )
        )

        # construct transition network
        self.layer_list.append(
            tf.keras.layers.Dense(
                trans_arch[1],
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                input_shape=(trans_arch[0],),
                name="transition_layer_1",
            )
        )
        for i in range(2, len(trans_arch) - 1):
            self.layer_list.append(
                tf.keras.layers.Dense(
                    trans_arch[i],
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    name=f"transition_layer_{i}",
                )
            )
        self.layer_list.append(
            tf.keras.layers.Dense(
                trans_arch[-1],
                activation="linear",
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                name=f"transition_layer_{len(trans_arch)-1}",
            )
        )

        # construct collision network
        self.layer_list.append(
            tf.keras.layers.Dense(
                col_arch[1],
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                input_shape=(col_arch[0],),
                name="collision_layer_1",
            )
        )
        for i in range(2, len(col_arch) - 1):
            self.layer_list.append(
                tf.keras.layers.Dense(
                    col_arch[i],
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    name=f"collision_layer_{i}",
                )
            )
        self.layer_list.append(
            tf.keras.layers.Dense(
                col_arch[-1],
                activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                name=f"collision_layer_{len(col_arch)-1}",
            )
        )

    def call(self, s_0, training=False):
        a = self.layer_list[0](s_0)
        for l in range(1, len(self.ctrl_arch) - 1):
            a = self.layer_list[l](a)
        s1 = self.layer_list[len(self.ctrl_arch) - 1](tf.concat((s_0, a), 1))
        for l in range(
            len(self.ctrl_arch), len(self.ctrl_arch) + len(self.trans_arch) - 2
        ):
            s1 = self.layer_list[l](s1)
        c = self.layer_list[len(self.ctrl_arch) + len(self.trans_arch) - 2](s1)
        for l in range(
            len(self.ctrl_arch) + len(self.trans_arch) - 1,
            len(self.ctrl_arch) + len(self.trans_arch) + len(self.col_arch) - 3,
        ):
            c = self.layer_list[l](c)
        return a, s1, c


def give_arch(model):
    arch = []
    for l in range(len(model.layers)):
        type_lay = len(model.layers[l].output_shape)
        if type_lay == 2:
            in_shape = model.layers[l].input_shape[1]
            out_shape = model.layers[l].output_shape[1]
        else:
            in_shape = model.layers[l].input_shape[0][1]
            out_shape = model.layers[l].output_shape[0][1]
        if l == 0:
            if in_shape != out_shape:
                arch.append(in_shape)

        arch.append(out_shape)

    return arch


def assign_weights(model_new, model_ctrl, model_trans, model_coll):
    weights_ctrl = model_ctrl.get_weights()
    weights_trans = model_trans.get_weights()
    weights_coll = model_coll.get_weights()
    for l in range(len(model_new.ctrl_arch) - 1):
        model_new.layer_list[l].set_weights(
            [weights_ctrl[2 * l], weights_ctrl[2 * l + 1]]
        )
    for l in range(len(model_new.trans_arch) - 1):
        model_new.layer_list[l + len(model_new.ctrl_arch) - 1].set_weights(
            [weights_trans[2 * l], weights_trans[2 * l + 1]]
        )
    for l in range(len(model_new.col_arch) - 1):
        model_new.layer_list[
            l + len(model_new.ctrl_arch) + len(model_new.trans_arch) - 2
        ].set_weights([weights_coll[2 * l], weights_coll[2 * l + 1]])


def combine_nets(
    model_ctrl,
    model_trans,
    model_coll,
    regularizer_rate=0.001,
):
    arch_model_ctrl = give_arch(model_ctrl)
    arch_model_trans = give_arch(model_trans)
    arch_model_coll = give_arch(model_coll)

    # construct model
    model = state_col_block(
        regularizer_rate, arch_model_ctrl, arch_model_trans, arch_model_coll
    )
    # build model
    _, _, _ = model(tf.random.uniform((1, arch_model_ctrl[0])))
    # assign weights
    assign_weights(model, model_ctrl, model_trans, model_coll)

    model.summary()

    return model


def inflate_hit(hit):
    for i in range(hit.shape[0] - 1):
        if hit[i] == 0 and hit[i + 1] == 1:
            hit[i] = 1
    return hit


def detect_idle_indices_of_robot(vel):
    # find the first index where the robot is not moving

    for i in range(vel.shape[0]):
        if vel[i, 0] != 0.0 or vel[i, 1] != 0.0:
            break
    # find the last index where the robot is not moving
    for j in range(vel.shape[0] - 1, 0, -1):
        if vel[j, 0] != 0:
            break
    return i, j + 5


def angular_diff(x, y):  # angular difference
    d = y - x
    return np.arctan2(np.sin(d), np.cos(d))  # type: ignore


def calc_distance_and_angle(from_node, to_node):
    dx = to_node - from_node
    d = np.hypot(dx[0], dx[1])  # type: ignore
    theta = np.arctan2(dx[1], dx[0])  # type: ignore
    return d, theta


def load_raw(read_dir, num_samples, num_inflate):
    poses = []
    scans = []
    vels = []
    hits = []
    for i in range(num_samples):
        read_path = read_dir + f"/sample{i+1}"
        poses.append(np.load(f"{read_path}/pos.npy", allow_pickle=True))
        scan = np.load(f"{read_path}/scans.npy", allow_pickle=True)
        # chang inf values if scan to 0.6
        scan[scan == np.inf] = 3.0  # type: ignore
        scans.append(scan)
        vels.append(np.load(f"{read_path}/vel.npy", allow_pickle=True))

        hit = np.load(f"{read_path}/hit.npy", allow_pickle=True)
        hit = hit.astype(np.int32)  # type: ignore
        hit = hit.reshape(-1, 1)
        for _ in range(num_inflate):
            hit = inflate_hit(hit)
            hit = inflate_hit(hit)
        hits.append(np.concatenate((1 - hit, hit), axis=1))

    return poses, scans, vels, hits


def retrieve_goal(goal, pose):
    dist_list = []
    for i in range(len(goal)):
        dist_list.append(np.linalg.norm(goal[i][:2] - pose[:2]))
    return goal[np.argmin(dist_list)]


def load_data(read_dir, num_samples, num_inflate, goal):
    poses, scans, vels, hits = load_raw(read_dir, num_samples, num_inflate)
    x = []
    y_ctrl = []
    y_trans = []
    y_col = []
    for i in range(num_samples):
        goal_pose = retrieve_goal(goal, poses[i][-1, :2])
        print(f"loading sample {i+1}, goal: {goal_pose}")
        traj_x = []
        traj_y_ctrl = []
        traj_y_trans = []
        traj_y_col = []
        ids = detect_idle_indices_of_robot(vels[i])
        plt.plot(poses[i][ids[0] : ids[1], 0], poses[i][ids[0] : ids[1], 1])
        for j in range(ids[0], ids[1]):
            dist, ang = calc_distance_and_angle(poses[i][j, :2], goal_pose)
            traj_x.append(
                np.concatenate(
                    (
                        goal_pose,
                        np.array(
                            [
                                dist,
                                angular_diff(ang, poses[i][j, 2]),
                            ]
                        ),
                        scans[i][j],
                    )
                )
            )
            traj_y_ctrl.append(vels[i][j])
            traj_y_trans.append(scans[i][j + 1])
            traj_y_col.append(hits[i][j + 1])
        x.append(np.array(traj_x))
        y_ctrl.append(np.array(traj_y_ctrl))
        y_trans.append(np.array(traj_y_trans))
        y_col.append(np.array(traj_y_col))

    plt.show()

    return x, y_ctrl, y_trans, y_col


def separate_train_test(x, y_ctrl, y_trans, y_col, test_ratio=0.2):
    x_train = []
    y_ctrl_train = []
    y_trans_train = []
    y_col_train = []
    x_test = []
    y_ctrl_test = []
    y_trans_test = []
    y_col_test = []
    num_train = int((1 - test_ratio) * len(x))
    num_test = len(x) - num_train
    # shuffle the data
    # idx = np.random.permutation(len(x))
    # train_idx = idx[:num_train]
    # test_idx = idx[num_train:]
    # for i in range(num_train):
    #     x_train.append(x[train_idx[i]])
    #     y_ctrl_train.append(y_ctrl[train_idx[i]])
    #     y_col_train.append(y_col[train_idx[i]])
    # for i in range(num_test):
    #     x_test.append(x[test_idx[i]])
    #     y_ctrl_test.append(y_ctrl[test_idx[i]])
    #     y_col_test.append(y_col[test_idx[i]])
    # idx = np.random.permutation(int(len(x) / 2))
    # train_idx = np.concatenate(
    #     (idx[: int(num_train / 2)], idx[: int(num_train / 2)] + int(len(x) / 2))
    # )
    # np.random.shuffle(train_idx)
    # test_idx = np.concatenate(
    #     (idx[int(num_train / 2) :], idx[int(num_train / 2) :] + int(len(x) / 2))
    # )
    num_scenarios = 6
    idx_list = [
        np.random.permutation(int(len(x) / num_scenarios))
        + i * int(len(x) / num_scenarios)
        for i in range(num_scenarios)
    ]
    # write the following another way
    train_idx = np.array([])

    for i in range(num_scenarios):
        train_idx = np.concatenate(
            (
                train_idx,
                idx_list[i][: int(num_train / num_scenarios)],
            )
        )

    test_idx = np.array([], dtype=int)
    for i in range(num_scenarios):
        test_idx = np.concatenate(
            (
                test_idx,
                idx_list[i][int(num_train / num_scenarios) :],
            )
        )
    # idx = np.random.permutation(int(len(x) / 6))
    # # train_idx = np.concatenate(
    # #     (idx[: int(num_train / 2)], idx[: int(num_train / 2)] + int(len(x) / 2))
    # # )
    # # np.random.shuffle(train_idx)
    # # test_idx = np.concatenate(
    # #     (idx[int(num_train / 2) :], idx[int(num_train / 2) :] + int(len(x) / 2))
    # # )
    # train_idx = idx[:num_train]
    # test_idx = idx[num_train:]
    # print(train_idx)
    # for i in range(num_train):
    #     x_train.append(x[train_idx[i]])
    #     y_train.append(y[train_idx[i]])
    # for i in range(num_test):
    #     x_test.append(x[test_idx[i]])
    #     y_test.append(y[test_idx[i]])

    # idx_1 = np.random.permutation(range(0, int(len(x) / 3)))
    # idx_2 = np.random.permutation(range(int(len(x) / 3), int(2 * len(x) / 3)))
    # idx_3 = np.random.permutation(range(int(2 * len(x) / 3), len(x)))
    # train_idx = np.concatenate(
    #     (
    #         idx_1[: int(num_train / 3)],
    #         idx_2[: int(num_train / 3)],
    #         idx_3[: int(num_train / 3)],
    #     )
    # )
    # test_idx = np.concatenate(
    #     (
    #         idx_1[int(num_train / 3) :],
    #         idx_2[int(num_train / 3) :],
    #         idx_3[int(num_train / 3) :],
    #     )
    # )
    # # shuffle train and test indices
    np.random.shuffle(train_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(train_idx)
    print(train_idx)
    for i in range(num_train):
        x_train.append(x[int(0.001 + train_idx[i])])
        y_ctrl_train.append(y_ctrl[int(0.001 + train_idx[i])])
        y_trans_train.append(y_trans[int(0.001 + train_idx[i])])
        y_col_train.append(y_col[int(0.001 + train_idx[i])])
    for i in range(num_test):
        x_test.append(x[int(0.001 + test_idx[i])])
        y_ctrl_test.append(y_ctrl[int(0.001 + test_idx[i])])
        y_trans_test.append(y_trans[int(0.001 + test_idx[i])])
        y_col_test.append(y_col[int(0.001 + test_idx[i])])

    return (
        x_train,
        y_ctrl_train,
        y_trans_train,
        y_col_train,
        x_test,
        y_ctrl_test,
        y_trans_test,
        y_col_test,
    )


def number_of_collisions(model, x):
    sum = 0
    for traj in x:
        out = model.predict(traj)[-1]
        sum += np.where(out[:, 1] > 0.5)[0].shape[0]
    return sum


def miniBatch(x, y_ctrl, y_trans, y_col, batchSize):
    numObs = x.shape[0]
    batches = []
    batchNum = int(np.floor(numObs / batchSize))

    if numObs % batchSize == 0:
        for i in range(batchNum):
            x_batch = x[i * batchSize : (i + 1) * batchSize, :]
            y_batch_ctrl = y_ctrl[i * batchSize : (i + 1) * batchSize, :]
            y_batch_trans = y_trans[i * batchSize : (i + 1) * batchSize, :]
            y_batch_col = y_col[i * batchSize : (i + 1) * batchSize, :]
            batches.append((x_batch, y_batch_ctrl, y_batch_trans, y_batch_col))
    else:
        for i in range(batchNum):
            x_batch = x[i * batchSize : (i + 1) * batchSize, :]
            y_batch_ctrl = y_ctrl[i * batchSize : (i + 1) * batchSize, :]
            y_batch_trans = y_trans[i * batchSize : (i + 1) * batchSize, :]
            y_batch_col = y_col[i * batchSize : (i + 1) * batchSize, :]
            batches.append((x_batch, y_batch_ctrl, y_batch_trans, y_batch_col))
        x_batch = x[batchNum * batchSize :, :]
        y_batch_ctrl = y_ctrl[batchNum * batchSize :, :]
        y_batch_trans = y_trans[batchNum * batchSize :, :]
        y_batch_col = y_col[batchNum * batchSize :, :]
        batches.append((x_batch, y_batch_ctrl, y_batch_trans, y_batch_col))
    return batches


def plot_model_control_out(model, x, color="blue", control=None, ax=None):
    ctrl = np.array([]).reshape(0, 2)
    coll = np.array([]).reshape(0, 2)
    ctrl_ref = np.array([]).reshape(0, 2)
    for i in range(len(x)):
        out = model.predict(x[i])
        ctrl = np.vstack((ctrl, out[0]))
        coll = np.vstack((coll, out[2]))
        if control is not None:
            ctrl_ref = np.vstack((ctrl_ref, control[i]))
    if ax is None:
        fig, ax = plt.subplots(2, 1)
    ax[0].plot(ctrl[:, 0], color=color, alpha=0.5, label="velocity")
    ax[1].plot(ctrl[:, 1], color=color, alpha=0.5, label="angle")
    if control is not None:
        ax[0].plot(ctrl_ref[:, 0], color="red", alpha=0.5, label="ref velocity")
        ax[1].plot(ctrl_ref[:, 1], color="red", alpha=0.5, label="ref angle")
    ax[0].legend()
    ax[1].legend()
    ax[0].scatter(
        np.where(coll[:, 1] > 0.5)[0],
        ctrl[np.where(coll[:, 1] > 0.5)[0], 0],
        color=color,
    )
    ax[1].scatter(
        np.where(coll[:, 1] > 0.5)[0],
        ctrl[np.where(coll[:, 1] > 0.5)[0], 1],
        color=color,
    )
    return ax
    # fill between the velocity and angles that collision[1] is above .5


def plot_loss_surface(control, state, model_coll, model_trans):
    pass


if __name__ == "__main__":
    policy_num = 13
    transition_num = 4
    collison_num = 2
    repaired_num = 33
    batch_size = 32  # 5 previously
    learning_rate = 0.00005
    environ = "hospital"
    weight_ctrl = 1.0
    weight_trans = 1.0
    weight_col = 1.0
    epochs = 100
    velocity_limit = 0.9
    # parameters for updating learning rate
    lr_decay_count = 0.0
    lr_decay_factor = 0.1
    lr_patience = 10.0
    loss_prev = 1000.0
    lr_min = 5e-5
    lr_loss_tol = 1e-3
    # constraint parameters
    # Set the initial values of mu_col and lambda
    mu_col = tf.constant(5.0, dtype="float64")
    lambda_col = tf.constant(0.0, dtype="float64")
    # nu_col = tf.cast(1.0 / mu_col, dtype="float64")
    nu_col = tf.cast(0.01, dtype="float64")
    kappa_col = tf.constant(10, dtype="float64")

    mu_vel = tf.constant(10.0, dtype="float64")
    lambda_vel = tf.constant(0.0, dtype="float64")
    # nu_vel = tf.cast(1.0 / mu_vel, dtype="float64")
    nu_vel = tf.cast(0.0001, dtype="float64")
    kappa_vel = tf.constant(5, dtype="float64")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)],
            )
        except RuntimeError as e:
            print(e)

    # get number of samples
    data_dir = (
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        + f"/data/{environ}/joy"
    )
    num_samples = len(os.listdir(data_dir))
    num_inflate = 5
    goal = [
        np.array([-10.0, 10]),
        np.array([-10.0, 5.0]),
        np.array([-9.0, -9.0]),
        np.array([10.0, 10.0]),
        np.array([10.0, 5.0]),
        np.array([9.0, -9.0]),
    ]
    x, y_ctrl, y_trans, y_col = load_data(data_dir, num_samples, num_inflate, goal)
    # train the model
    (
        x_train,
        y_ctrl_train,
        y_trans_train,
        y_col_train,
        x_test,
        y_ctrl_test,
        y_trans_test,
        y_col_test,
    ) = separate_train_test(x, y_ctrl, y_trans, y_col, 0.2)
    # hand-label the collision data
    for traj in y_col:
        for i in range(traj.shape[0]):
            if traj[i, 0] == 0:
                traj[i, 0] = 1
                traj[i, 1] = 0
    # specify the velocity of the robot to be zero when dist < 0.5
    # for i in range(len(x)):
    #     for j in range(x[i].shape[0]):
    #         if x[i][j, 0] < 0.5:
    #             y_ctrl[i][j, 0] = 0.0
    # load network
    model_ctrl = keras.models.load_model(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        + f"/trained_models/{environ}/basic/policy/model_{policy_num}"
    )
    model_trans = keras.models.load_model(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        + f"/trained_models/{environ}/basic/transition/model_{transition_num}"
    )
    model_coll = keras.models.load_model(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        + f"/trained_models/{environ}/basic/collision/model_{collison_num}"
    )

    model = combine_nets(model_ctrl, model_trans, model_coll)
    # plot before repair
    ax = plot_model_control_out(model, x, color="blue", control=y_ctrl)
    # plt.show()

    print(
        f"number of collisions in training set: {number_of_collisions(model, x_train)}"
    )
    print(f"number of collisions in test set: {number_of_collisions(model, x_test)}")
    model.layers[0].trainable = True
    model.layers[1].trainable = True
    model.layers[2].trainable = True
    model.layers[3].trainable = False
    model.layers[4].trainable = False
    model.layers[5].trainable = False
    model.layers[6].trainable = False
    model.layers[7].trainable = False
    model.layers[8].trainable = False
    model.predict(x_train[0])

    def loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def penalty_col(y):
        # return tf.reduce_sum(tf.square(tf.nn.relu(0.8 - y[:, 0])))
        return tf.reduce_sum(tf.square(y[:, 1]))
        # return tf.reduce_sum(tf.square(tf.nn.relu(0.9 + y[:, 1] - y[:, 0])))

    def constraint_col(y):
        # return tf.reduce_sum(tf.nn.relu(0.8 - y[:, 0]))
        return tf.reduce_sum(y[:, 1])
        # return tf.reduce_sum(tf.nn.relu(0.9 + y[:, 1] - y[:, 0]))

    def constraint_vel(y):
        return tf.reduce_sum(tf.nn.relu(y[:, 0] - velocity_limit))

    def penalty_vel(y):
        return tf.reduce_sum(tf.square(tf.nn.relu(y[:, 0] - velocity_limit)))

    def augmented_lagrangian(
        x, y_true_ctrl, y_true_trans, y_true_col, mu_col, lambda_col, mu_vel, lambda_vel
    ):
        y_pred = model(x)
        return (
            100 * loss(y_true_ctrl, tf.cast(y_pred[0], dtype=y_true_ctrl.dtype))
            + 1 * loss(y_true_trans, tf.cast(y_pred[1], dtype=y_true_trans.dtype))
            - tf.cast(
                lambda_col * constraint_col(tf.cast(y_pred[-1], tf.float64)),
                dtype=y_true_col.dtype,
            )
            + tf.cast(
                mu_col / 2 * penalty_col(tf.cast(y_pred[-1], tf.float64)),
                dtype=y_true_col.dtype,
            )
            - tf.cast(
                lambda_vel * constraint_vel(tf.cast(y_pred[0], tf.float64)),
                dtype=y_true_col.dtype,
            )
            + tf.cast(
                mu_vel / 2 * penalty_vel(tf.cast(y_pred[0], tf.float64)),
                dtype=y_true_col.dtype,
            )
        )

    # Define the training loop
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(
        x, y_true_ctrl, y_true_trans, y_true_col, mu_col, lambda_col, mu_vel, lambda_vel
    ):
        x = tf.convert_to_tensor(x, dtype=model.trainable_variables[0].dtype)
        y_true_ctrl = tf.convert_to_tensor(
            y_true_ctrl, dtype=model.trainable_variables[0].dtype
        )
        y_true_trans = tf.convert_to_tensor(
            y_true_trans, dtype=model.trainable_variables[0].dtype
        )
        y_true_col = tf.convert_to_tensor(
            y_true_col, dtype=model.trainable_variables[0].dtype
        )
        with tf.GradientTape() as tape:
            loss_value = augmented_lagrangian(
                x,
                y_true_ctrl,
                y_true_trans,
                y_true_col,
                mu_col,
                lambda_col,
                mu_vel,
                lambda_vel,
            )
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value

    # check trainable layers
    for layer in model.layers:
        print("trainable layer: ", layer.name, layer.trainable)
    ww0 = model.layers[0].get_weights()[0]
    ww3 = model.layers[3].get_weights()[0]

    # Train model

    # chnge float type of x_train
    x_train = [x.astype("float32") for x in x_train]
    y_ctrl_train = [y.astype("float32") for y in y_ctrl_train]
    y_trans_train = [y.astype("float32") for y in y_trans_train]
    y_col_train = [y.astype("float32") for y in y_col_train]

    # Train the model with constraints using the augmented Lagrangian method
    # for var in model.trainable_variables:
    #     var = tf.cast(var, tf.float64)
    # id = 1

    # specify test variables
    inp_test = np.vstack(x_test)
    out_ctrl_test = np.vstack(y_ctrl_test)
    out_trans_test = np.vstack(y_trans_test)
    out_col_test = np.vstack(y_col_test)

    # specify train variable
    inp_train = np.vstack(x_train)
    out_ctrl_train = np.vstack(y_ctrl_train)
    out_trans_train = np.vstack(y_trans_train)
    out_col_train = np.vstack(y_col_train)

    # remove input constraint violating points
    id = []
    coll_pred_input = model_coll.predict(inp_train[:, 4:])
    for i in range(coll_pred_input.shape[0]):
        if coll_pred_input[i, 1] > 0.5:
            id.append(i)

    inp_train = np.delete(inp_train, id, axis=0)
    out_ctrl_train = np.delete(out_ctrl_train, id, axis=0)
    out_trans_train = np.delete(out_trans_train, id, axis=0)
    out_col_train = np.delete(out_col_train, id, axis=0)

    id = []
    coll_pred_input = model_coll.predict(inp_test[:, 4:])
    for i in range(coll_pred_input.shape[0]):
        if coll_pred_input[i, 1] > 0.5:
            id.append(i)

    inp_test = np.delete(inp_test, id, axis=0)
    out_ctrl_test = np.delete(out_ctrl_test, id, axis=0)
    out_trans_test = np.delete(out_trans_test, id, axis=0)
    out_col_test = np.delete(out_col_test, id, axis=0)

    # create batches from train data
    batches = miniBatch(
        inp_train, out_ctrl_train, out_trans_train, out_col_train, batch_size
    )
    # out_ctrl_train = tf.data.Dataset.from_tensor_slices(out_ctrl_train).batch(1)

    # track losses
    history = {
        "training loss": {
            "ctrl": [],
            "trans": [],
            "col": [],
            "const_col": [],
            "const_vel": [],
            "lambda": [lambda_col],
        },
        "validation loss": {
            "ctrl": [],
            "trans": [],
            "col": [],
            "const_col": [],
            "const_vel": [],
        },
    }
    save_loss = 1000.0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in batches:
            batch_loss = train_step(
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                mu_col,
                lambda_col,
                mu_vel,
                lambda_vel,
            )
            epoch_loss += batch_loss
        epoch_loss /= len(x_train)

        ## evaluate metrics
        # training metrics
        pred_ctrl, pred_trans, pred_col = model(inp_train)
        history["training loss"]["ctrl"].append(loss(out_ctrl_train, pred_ctrl).numpy())
        history["training loss"]["trans"].append(
            loss(out_trans_train, pred_trans).numpy()
        )
        history["training loss"]["col"].append(loss(out_col_train, pred_col).numpy())
        history["training loss"]["const_col"].append(constraint_col(pred_col).numpy())
        history["training loss"]["const_vel"].append(constraint_vel(pred_ctrl).numpy())

        # validation metrics
        pred_ctrl, pred_trans, pred_col = model(inp_test)
        history["validation loss"]["ctrl"].append(
            loss(out_ctrl_test, pred_ctrl).numpy()
        )
        history["validation loss"]["trans"].append(
            loss(out_trans_test, pred_trans).numpy()
        )
        history["validation loss"]["col"].append(loss(out_col_test, pred_col).numpy())
        history["validation loss"]["const_col"].append(constraint_col(pred_col).numpy())
        history["validation loss"]["const_vel"].append(
            constraint_vel(pred_ctrl).numpy()
        )

        ## update mu_col and lambda
        # if history["training loss"]["const_col"][-1] <= nu_col:
        #     lambda_col = tf.cast(
        #         lambda_col - mu_col * history["training loss"]["const_col"][-1], dtype="float64"
        #     )
        #     nu_col = tf.cast(nu_col / mu_col, dtype="float64")
        # else:
        #     mu_col = tf.cast(mu_col * 1.0, dtype="float64")
        if (epoch + 1) % 10 == 0:
            lambda_col = tf.cast(
                lambda_col - nu_col * history["training loss"]["const_col"][-1],
                dtype="float64",
            )
            mu_col = tf.cast(mu_col * kappa_col, dtype="float64")
            history["training loss"]["lambda"].append(lambda_col.numpy())

            lambda_vel = tf.cast(
                lambda_vel - nu_vel * history["training loss"]["const_vel"][-1],
                dtype="float64",
            )
            mu_vel = tf.cast(mu_vel * kappa_vel, dtype="float64")

        # update learning rate
        if loss_prev - history["validation loss"]["const_col"][-1] > lr_loss_tol:
            # lr_decay_count = 0.0
            pass
        else:
            lr_decay_count += 1.0
            if lr_decay_count >= lr_patience:
                lr_decay_count = 0.0
                new_lr = optimizer.learning_rate * lr_decay_factor
                if new_lr.numpy() >= lr_min:
                    optimizer.learning_rate.assign(new_lr)

        print(loss_prev - history["validation loss"]["const_col"][-1])
        print(lr_decay_count)
        loss_prev = history["validation loss"]["const_col"][-1]

        print(
            "epoch %d: loss_ctrl=%f, loss_trans=%f, loss_col=%f, loss_const_col=%f, loss_const_vel=%f, lambda_col=%f, mu_col=%f, lambda_vel=%f, mu_vel=%f"
            % (
                epoch,
                history["training loss"]["ctrl"][-1],
                history["training loss"]["trans"][-1],
                history["training loss"]["col"][-1],
                history["training loss"]["const_col"][-1],
                history["training loss"]["const_vel"][-1],
                lambda_col,
                mu_col,
                lambda_vel,
                mu_vel,
            )
        )
        print(
            "          loss_ctrl_val=%f, loss_trans_val=%f, loss_col_val=%f, loss_const_col_val=%f, loss_const_vel_val=%f"
            % (
                history["validation loss"]["ctrl"][-1],
                history["validation loss"]["trans"][-1],
                history["validation loss"]["col"][-1],
                history["validation loss"]["const_col"][-1],
                history["validation loss"]["const_vel"][-1],
            )
        )
        print(f"          learning rate: {optimizer.learning_rate.numpy()}")
        # save model
        if save_loss > history["validation loss"]["const_col"][-1]:
            # change the model weights of policy
            id_weigts = 0
            for l in range(len(model_ctrl.layers)):
                if (len(model_ctrl.layers[l].get_weights())) > 0:
                    model_ctrl.layers[l].set_weights(
                        model.layers[id_weigts].get_weights()
                    )
                    id_weigts += 1

            if (
                not os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                + f"/trained_models/{environ}/basic/repaired"
            ):
                os.makedirs(
                    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                    + f"/trained_models/{environ}/basic/repaired"
                )
            tf.keras.models.save_model(
                model_ctrl,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                + f"/trained_models/{environ}/basic/repaired/model_{repaired_num}",
                overwrite=True,
                include_optimizer=False,
                save_format=None,
                signatures=None,
                options=None,
                save_traces=True,
            )
            save_loss = history["validation loss"]["const_col"][-1]
            print("checkpoint saved")
        # call learning rate callback
        # lr_callback.on_epoch_end(epoch, {"val_constraint": constraint_value})
    print(
        f"number of collisions in training set: {number_of_collisions(model, x_train)}"
    )
    print(f"number of collisions in test set: {number_of_collisions(model, x_test)}")
    print("ww0 error: ", np.linalg.norm(ww0 - model.layers[0].get_weights()[0]))
    print("ww3 error: ", np.linalg.norm(ww3 - model.layers[3].get_weights()[0]))
    # plot signals after repair
    plot_model_control_out(model, x, color="k", control=None, ax=ax)
    plt.show()

    for traj, ctrl in zip(x_test[:1], y_ctrl_test[:1]):
        pred_control = model.predict(traj)[0]
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(pred_control[:, 0], label="pred", color="crimson")
        axs[0].plot(ctrl[:, 0], label="true", color="blue")
        axs[1].plot(pred_control[:, 1], label="pred", color="crimson")
        axs[1].plot(ctrl[:, 1], label="true", color="blue")
    plt.show()
    # make subplots
    fig, axs = plt.subplots(5, 1)
    color_hist = "k"
    axs[0].plot(
        history["training loss"]["ctrl"],
        label="ctrl",
        color=color_hist,
        linewidth=2,
        linestyle="-",
    )
    axs[0].plot(
        history["validation loss"]["ctrl"],
        label="ctrl_val",
        color=color_hist,
        linewidth=2,
        linestyle="--",
    )
    axs[1].plot(
        history["training loss"]["trans"],
        label="trans",
        color=color_hist,
        linewidth=2,
        linestyle="-",
    )
    axs[1].plot(
        history["validation loss"]["trans"],
        label="trans_val",
        color=color_hist,
        linewidth=2,
        linestyle="--",
    )
    axs[2].plot(
        history["training loss"]["const_col"],
        label="const_col",
        color=color_hist,
        linewidth=2,
        linestyle="-",
    )
    axs[2].plot(
        history["validation loss"]["const_col"],
        label="const_col_val",
        color=color_hist,
        linewidth=2,
        linestyle="--",
    )
    axs[3].plot(
        history["training loss"]["const_vel"],
        label="const_vel",
        color=color_hist,
        linewidth=2,
        linestyle="-",
    )
    axs[3].plot(
        history["validation loss"]["const_vel"],
        label="const_vel_val",
        color=color_hist,
        linewidth=2,
        linestyle="--",
    )
    axs[4].plot(
        history["training loss"]["lambda"],
        label="lambda",
        color=color_hist,
        linewidth=2,
        linestyle="-",
    )
    axs[0].set_ylabel("control loss")
    axs[1].set_ylabel("transition loss")
    axs[2].set_ylabel("collision constraint loss")
    axs[3].set_ylabel("velocity constraint loss")
    axs[4].set_ylabel("lambda")
    axs[4].set_xlabel("epochs")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()
    plt.show()

    # # change the model weights of policy
    # id_weigts = 0
    # for l in range(len(model_ctrl.layers)):
    #     if (len(model_ctrl.layers[l].get_weights())) > 0:
    #         model_ctrl.layers[l].set_weights(model.layers[id_weigts].get_weights())
    #         id_weigts += 1

    # if (
    #     not os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #     + f"/trained_models/{environ}/basic/repaired"
    # ):
    #     os.makedirs(
    #         os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #         + f"/trained_models/{environ}/basic/repaired"
    #     )
    # # save model
    # tf.keras.models.save_model(
    #     model_ctrl,
    #     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #     + f"/trained_models/{environ}/basic/repaired/model_{repaired_num}",
    #     overwrite=True,
    #     include_optimizer=False,
    #     save_format=None,
    #     signatures=None,
    #     options=None,
    #     save_traces=True,
    # )
    # # log train
    # logfile_direc = (
    #     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #     + f"/log_models/{environ}/basic/repaired_log_model_{repaired_num}"
    # )
    # checkpoint_directory = (
    #     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #     + f"/log_models/{environ}/basic/repaired_checkpoint_model_{repaired_num}"
    # )
    # if not os.path.exists(checkpoint_directory):
    #     checkpoint_directory
    # if not os.path.exists(logfile_direc):
    #     os.makedirs(logfile_direc)
    # inp_test = np.vstack(x_test)
    # out_ctrl_test = np.vstack(y_ctrl_test)
    # out_trans_test = np.vstack(y_trans_test)
    # out_col_test = np.vstack(y_col_test)
    # for i in range(len(x_train)):
    #     filepath_check = f"{checkpoint_directory}/sample_{i+1}"
    #     filepath_log = f"{logfile_direc}/sample_{i+1}"
    #     print(f"training sample {i+1}")
    #     inp_train = np.vstack(x_train[: i + 1])
    #     out_ctrl_train = np.vstack(y_ctrl_train[: i + 1])
    #     out_trans_train = np.vstack(y_trans_train[: i + 1])
    #     out_col_train = np.vstack(y_col_train[: i + 1])
    #     model.compile(
    #         optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    #         loss=[loss_ctrl, loss_trans, loss_coll],
    #         metrics=["accuracy"],
    #     )
    #     model.fit(
    #         inp_train,
    #         [out_ctrl_train, out_trans_train, out_col_train],
    #         validation_data=(inp_test, [out_ctrl_test, out_trans_test, out_col_test]),
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         use_multiprocessing=True,
    #         verbose=1,
    #         callbacks=[
    #             keras.callbacks.ModelCheckpoint(
    #                 filepath_check,
    #                 monitor="output_3_accuracy",
    #                 verbose=0,
    #                 save_best_only=True,
    #                 save_weight_only=False,
    #                 mode="auto",
    #                 save_freq="epoch",
    #                 options=None,
    #             ),
    #             keras.callbacks.EarlyStopping(
    #                 monitor="output_3_accuracy",
    #                 patience=10,
    #                 restore_best_weights=True,
    #             ),
    #             keras.callbacks.ReduceLROnPlateau(
    #                 monitor="output_3_accuracy",
    #                 factor=0.1,
    #                 patience=5,
    #                 verbose=1,
    #                 min_lr=0.0001,
    #             ),
    #             keras.callbacks.TensorBoard(log_dir=filepath_log),
    #         ],
    #     )
    #     print("ww0 error: ", np.linalg.norm(ww0 - model.layers[0].get_weights()[0]))
    #     print("ww3 error: ", np.linalg.norm(ww3 - model.layers[3].get_weights()[0]))
    #     epochs += 1

    # print("model loaded")

    # # change the model weights of policy
    # id_weigts = 0
    # for l in range(len(model_ctrl.layers)):
    #     if (len(model_ctrl.layers[l].get_weights())) > 0:
    #         model_ctrl.layers[l].set_weights(model.layers[id_weigts].get_weights())
    #         id_weigts += 1

    # if (
    #     not os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #     + f"/trained_models/{environ}/basic/repaired"
    # ):
    #     os.makedirs(
    #         os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #         + f"/trained_models/{environ}/basic/repaired"
    #     )
    # # save model
    # tf.keras.models.save_model(
    #     model_ctrl,
    #     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #     + f"/trained_models/{environ}/basic/repaired/model_{repaired_num}",
    #     overwrite=True,
    #     include_optimizer=False,
    #     save_format=None,
    #     signatures=None,
    #     options=None,
    #     save_traces=True,
    # )


# block = state_col_block(0.001)
# _ = block(tf.random.uniform((1, 7)))
