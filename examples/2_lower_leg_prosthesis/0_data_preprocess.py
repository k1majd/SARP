import os
import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # walking_ranges = np.array([[20710, 23460],[7185, 9640], [9995, 11150]])
    # walking_ranges = np.array([[9995, 11150], [11500, 12530],[7185, 9640]])
    walking_ranges = np.array([[20710+1000, 20710+2000],[20710+100, 20710+1000], [20710+2000, 23460]])
    # walking_ranges = np.array([[24303+1000, 24303+2000],[24303, 24303+1000], [24303+2000, 27993]])

    data_dir = os.path.dirname(os.path.realpath(__file__)) + f"/data/expert_data/full_walk_data1.csv"
    with open(data_dir) as csv_file:
        data = np.asarray(
            list(csv.reader(csv_file, delimiter=",")), dtype=np.float32
        )
    
    

    # data = pd.read_csv(data_dir)
    dankle = data[:, 37]
    dankle = dankle.reshape((dankle.shape[0], 1))
    dfemur_dtibia = data[:, [14, 18, 7, 11]]  # femur angle, vel, tibia angle, vel
    force = data[:, [21, 22, 23, 24]]

    # ankle angle
    dankle_train = dankle[walking_ranges[0, 0]:walking_ranges[0, 1]]
    dankle_validation = dankle[walking_ranges[1, 0]:walking_ranges[1, 1]]
    dankle_test = dankle[walking_ranges[2, 0]:walking_ranges[2, 1]]
    
    # femur and tibia angle
    dfemur_dtibia_train = (
        dfemur_dtibia[walking_ranges[0, 0]:walking_ranges[0, 1]]
        - dfemur_dtibia[walking_ranges[0, 0]:walking_ranges[0, 1]].mean(0)
        )/dfemur_dtibia[walking_ranges[0, 0]:walking_ranges[0, 1]].std(0)
    dfemur_dtibia_validation = (
        dfemur_dtibia[walking_ranges[1, 0]:walking_ranges[1, 1]]
        - dfemur_dtibia[walking_ranges[1, 0]:walking_ranges[1, 1]].mean(0)
        )/dfemur_dtibia[walking_ranges[1, 0]:walking_ranges[1, 1]].std(0)
    dfemur_dtibia_test = (
        dfemur_dtibia[walking_ranges[2, 0]:walking_ranges[2, 1]]
        - dfemur_dtibia[walking_ranges[2, 0]:walking_ranges[2, 1]].mean(0)
        )/dfemur_dtibia[walking_ranges[2, 0]:walking_ranges[2, 1]].std(0)
    
    # force
    # force_train = (
    #     force[walking_ranges[0, 0]:walking_ranges[0, 1]]
    #     - force[walking_ranges[0, 0]:walking_ranges[0, 1]].mean(0)
    #     )/force[walking_ranges[0, 0]:walking_ranges[0, 1]].std(0)
    # force_validation = (
    #     force[walking_ranges[1, 0]:walking_ranges[1, 1]]
    #     - force[walking_ranges[1, 0]:walking_ranges[1, 1]].mean(0)
    #     )/force[walking_ranges[1, 0]:walking_ranges[1, 1]].std(0)
    # force_test = (
    #     force[walking_ranges[2, 0]:walking_ranges[2, 1]]
    #     - force[walking_ranges[2, 0]:walking_ranges[2, 1]].mean(0)
    #     )/force[walking_ranges[2, 0]:walking_ranges[2, 1]].std(0)
    force_train =force[walking_ranges[0, 0]:walking_ranges[0, 1]]
    force_validation = force[walking_ranges[1, 0]:walking_ranges[1, 1]]
    force_test = force[walking_ranges[2, 0]:walking_ranges[2, 1]]
    

    # save train data 
    current_direc = os.path.dirname(os.path.realpath(__file__))
    save_folder = current_direc + "/data/expert_data/sample1"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    np.save(save_folder + "/dankle_train.npy", dankle_train)
    np.save(save_folder + "/dfemur_dtibia_train.npy", dfemur_dtibia_train)
    np.save(save_folder + "/force_train.npy", force_train)

    # save validation data
    np.save(save_folder + "/dankle_validation.npy", dankle_validation)
    np.save(save_folder + "/dfemur_dtibia_validation.npy", dfemur_dtibia_validation)
    np.save(save_folder + "/force_validation.npy", force_validation)

    # save test data
    np.save(save_folder + "/dankle_test.npy", dankle_test)
    np.save(save_folder + "/dfemur_dtibia_test.npy", dfemur_dtibia_test)
    np.save(save_folder + "/force_test.npy", force_test)
