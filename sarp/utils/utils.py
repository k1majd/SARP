import numpy as np
from tensorflow import keras
import tensorflow as tf


class RepairModel(tf.keras.Model):
    def __init__(
        self,
        regularizer_rate,
        policy_arch,
        predictive_arch,
        activ_policy,
        activ_predictive,
    ):
        super(RepairModel, self).__init__()
        self.layer_list = []
        self.policy_arch = policy_arch
        self.predictive_arch = predictive_arch

        # construct policy network
        self.layer_list.append(
            tf.keras.layers.Dense(
                policy_arch[1],
                activation=activ_policy[0],
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                input_shape=(policy_arch[0],),
                dtype=tf.float32,
                name="policy_layer_1",
            )
        )
        for i in range(2, len(policy_arch) - 1):
            self.layer_list.append(
                tf.keras.layers.Dense(
                    policy_arch[i],
                    activation=activ_policy[i - 1],
                    kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    dtype=tf.float32,
                    name=f"policy_layer_{i}",
                )
            )
        self.layer_list.append(
            tf.keras.layers.Dense(
                policy_arch[-1],
                activation=activ_policy[-1],
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                dtype=tf.float32,
                name=f"policy_layer_{len(policy_arch)-1}",
            )
        )

        # construct predictive network
        self.layer_list.append(
            tf.keras.layers.Dense(
                predictive_arch[1],
                activation=activ_predictive[0],
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                input_shape=(predictive_arch[0],),
                dtype=tf.float32,
                name="Predictive_layer_1",
            )
        )
        for i in range(2, len(predictive_arch) - 1):
            self.layer_list.append(
                tf.keras.layers.Dense(
                    predictive_arch[i],
                    activation=activ_predictive[i - 1],
                    kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                    dtype=tf.float32,
                    name=f"Predictive_layer_{i}",
                )
            )
        self.layer_list.append(
            tf.keras.layers.Dense(
                predictive_arch[-1],
                activation=activ_predictive[-1],
                kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                dtype=tf.float32,
                name=f"Predictive_layer_{len(predictive_arch)-1}",
            )
        )

    def call(self, s_0, training=False):
        a = self.layer_list[0](s_0)
        for l in range(1, len(self.policy_arch) - 1):
            a = self.layer_list[l](a)
        s1 = self.layer_list[len(self.policy_arch) - 1](tf.concat((s_0, a), 1))
        for l in range(
            len(self.policy_arch), len(self.policy_arch) + len(self.predictive_arch) - 2
        ):
            s1 = self.layer_list[l](s1)
        return a, s1


def inflate_hit(hit):
    for i in range(hit.shape[0] - 1):
        if hit[i] == 0 and hit[i + 1] == 1:
            hit[i] = 1
    return hit


def retrieve_goal(goal, pose):
    dist_list = []
    for i in range(len(goal)):
        dist_list.append(np.linalg.norm(goal[i][:2] - pose[:2]))
    return goal[np.argmin(dist_list)]


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


def calc_distance_and_angle(from_node, to_node):
    dx = to_node - from_node
    d = np.hypot(dx[0], dx[1])  # type: ignore
    theta = np.arctan2(dx[1], dx[0])  # type: ignore
    return d, theta


def angular_diff(x, y):  # angular difference
    d = y - x
    return np.arctan2(np.sin(d), np.cos(d))  # type: ignore


def load_raw(read_dir, num_samples, num_inflate=5):
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


def load_expert_data_hospital(read_dir, num_samples, num_inflate=5):
    goal = [
        np.array([-10.0, 10]),
        np.array([-10.0, 5.0]),
        np.array([-9.0, -9.0]),
        np.array([10.0, 10.0]),
        np.array([10.0, 5.0]),
        np.array([9.0, -9.0]),
    ]
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
        # plt.plot(poses[i][ids[0] : ids[1], 0], poses[i][ids[0] : ids[1], 1])
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

    # plt.show()

    return x, y_ctrl, y_trans, y_col


def give_arch(model):
    arch = []
    activation = []
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
        activation.append(model.layers[l].activation.__name__)

    return arch, activation


def assign_weights(model_new, policy, predictive):
    weights_ctrl = policy.get_weights()
    weights_trans = predictive.get_weights()
    for l in range(len(model_new.policy_arch) - 1):
        model_new.layer_list[l].set_weights(
            [weights_ctrl[2 * l], weights_ctrl[2 * l + 1]]
        )
    for l in range(len(model_new.predictive_arch) - 1):
        model_new.layer_list[l + len(model_new.policy_arch) - 1].set_weights(
            [weights_trans[2 * l], weights_trans[2 * l + 1]]
        )


def combine_nets(
    policy,
    predictive,
    regularizer_rate=0.001,
):
    arch_model_policy, activ_policy = give_arch(policy)
    arch_model_predictive, activ_predictive = give_arch(predictive)

    # construct model
    model = RepairModel(
        regularizer_rate,
        arch_model_policy,
        arch_model_predictive,
        activ_policy,
        activ_predictive,
    )
    # build model
    _, _ = model(tf.random.uniform((1, arch_model_policy[0]), dtype=tf.float32))
    # assign weights
    assign_weights(model, policy, predictive)

    model.summary()

    return model


def load_transition_data_hospital(read_dir, num_samples):
    goal = [
        np.array([-10.0, 10]),
        np.array([-10.0, 5.0]),
        np.array([-9.0, -9.0]),
        np.array([10.0, 10.0]),
        np.array([10.0, 5.0]),
        np.array([9.0, -9.0]),
    ]
    pos, scans, vel, _ = load_raw(read_dir, num_samples)
    # x should include [distance to goal, angle to goal, scan data]
    x = []
    # y should include [linear velocity, angular velocity]
    y = []
    for i in range(num_samples):
        goal_pose = retrieve_goal(goal, pos[i][-1, :2])
        print(f"Sample: {i+1}, goal: {goal_pose}")
        traj_x = []
        traj_y = []
        ids = detect_idle_indices_of_robot(vel[i])
        for j in range(ids[0], ids[1]):
            dist, ang = calc_distance_and_angle(pos[i][j, :2], goal_pose)
            inp = np.concatenate(
                (
                    goal_pose,
                    np.array(
                        [
                            dist,
                            angular_diff(ang, pos[i][j, 2]),
                        ]
                    ),
                    scans[i][j],
                )
            )
            traj_x.append(
                np.concatenate(
                    (
                        inp,
                        vel[i][j],
                    )
                )
            )
            traj_y.append(scans[i][j + 1])
        x.append(np.array(traj_x))
        y.append(np.array(traj_y))

    return x, y


def process_scan(scan):
    scan_list = []
    for i in range(scan.shape[0]):
        if len(scan[i]) < 10:
            scan_list.append(
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ]
            )
        else:
            scan_list.append(scan[i])

    return np.array(scan_list)


def load_wander_data_hospital(read_dir, num_samples, num_inflate=5):
    hits = []
    scans = []
    for i in range(num_samples):
        scan = np.load(read_dir + f"/sample{i+1}/scans.npy", allow_pickle=True)
        scan = process_scan(scan)
        # scan = scan.astype(np.float32)
        scan[scan == np.inf] = 3.0
        hit = np.load(read_dir + f"/sample{i+1}/hit.npy", allow_pickle=True)
        hit = hit.astype(np.int32)
        hit = hit.reshape(-1, 1)
        for j in range(num_inflate):
            hit = inflate_hit(hit)
            hit = inflate_hit(hit)
        idx_start = 0
        idx_end = hit.shape[0]
        search = False
        for j in range(hit.shape[0] - 1):
            if hit[j] == 0 and hit[j + 1] == 1:
                search = True
                gap_0 = j - idx_start + 1

            if search and hit[j] == 1 and hit[j + 1] == 0:
                gap_1 = j - gap_0 - idx_start + 1
                if gap_0 > gap_1:
                    idx_end = j + 1
                else:
                    idx_end = idx_start + 2 * gap_0
                search = False
                hit_arr = np.concatenate(
                    (1 - hit[idx_start:idx_end], hit[idx_start:idx_end]), axis=1
                )
                hits.append(hit_arr)
                scans.append(scan[idx_start:idx_end])
                idx_start = j + 1
        if search == True:
            hit_arr = np.concatenate(
                (1 - hit[idx_start : hit.shape[0]], hit[idx_start : hit.shape[0]]),
                axis=1,
            )
            hits.append(hit_arr)
            scans.append(scan[idx_start : hit.shape[0]])
    return hits, scans


def separate_train_test(data, test_ratio=0.2):
    num_train = int((1 - test_ratio) * len(data[0]))
    num_test = len(data[0]) - num_train

    num_scenarios = 6
    idx_list = [
        np.random.permutation(int(len(data[0]) / num_scenarios))
        + i * int(len(data[0]) / num_scenarios)
        for i in range(num_scenarios)
    ]
    # write the following another way
    train_idx = np.array([], dtype=int)

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

    np.random.shuffle(train_idx)
    train_data = []
    test_data = []
    for d in data:
        train_data.append([d[i] for i in train_idx])
        test_data.append([d[i] for i in test_idx])

    return train_data, test_data


def curriculum_training(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
):
    inp_test = np.vstack(x_test)
    out_test = np.vstack(y_test)
    epochs = epochs // len(x_train)
    for i in range(len(x_train)):
        print(f"Training on {i+1} sample(s)")
        inp_train = np.vstack(x_train[: i + 1])
        out_train = np.vstack(y_train[: i + 1])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mse"],
        )
        tf_callback = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=5,
                min_lr=0.0001,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
        ]
        model.fit(
            inp_train,
            out_train,
            validation_data=(inp_test, out_test),
            batch_size=batch_size,
            epochs=epochs,
            use_multiprocessing=True,
            verbose=1,
            shuffle=False,
            callbacks=tf_callback,
        )
        epochs += 1
