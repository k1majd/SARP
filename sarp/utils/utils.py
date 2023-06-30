import numpy as np
from tensorflow import keras


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


def load_data(read_dir, num_samples, num_inflate=5):
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
