
import numpy as np
import matplotlib.pyplot as plt 

# data = np.array([[1, 0, 1], [2,0,1], [-5, 0, -1]])
data = np.array([[2, 0, 1], [4,0,1], [-3, 0, -1]])
# data = np.array([[6, 2, 1], [3,4,-1], [7, -5, -1], [2,6,-1], [0,0, -1]])
# data2 = np.array([[5, 8, 1], [8,8,1], [9, 11, -1], [1.5,1.8,-1], [1, 2, -1]])
# data3 = np.array([[1, 1/2, 1], [0,1,1], [3/2, 2, 1], [2,-2,1], [-2, -2, -1]])


def svm_train_brute(training_data):
    positive = training_data[training_data[:, 2] == 1]
    negative = training_data[training_data[:, 2] == -1]
    margin = -9999999
    s_last, w_last, b_last = None, None, None
    list_of_w_and_b = []
    for pos in positive:
        for neg in negative:
            # mid_point = (pos[0:2] + neg[0:2]) / 2
            # print('mid_point', mid_point)
            # w = np.array(pos[:-1] - neg[:-1])
            # print('w', w)
            # w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
            # print('w_unit_vector', w)
            # b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
            # print('bias term', b)
            # list_of_w_and_b.append(w)
                        
            w, b = perpendicular_bisector_of_two_points(pos, neg)

            plot_hyper_binary(w, b, data)
            if margin <= compute_margin(training_data, w, b):
                print('last_margin_means_minimum_distance')
                margin = compute_margin(training_data, w, b)
                s_last = np.array([pos, neg])
                w_last = w
                b_last = b
            # plot_hyper_binary(w_last, b_last, data)
    # one positive - two negative sv
    # for neg in negative:
    #     for neg1 in negative:
    #         for pos in positive:
    #             if neg[0] != neg1[0] and neg[1] != neg1[1]:
    #                 separator = (neg1[:2] - neg[:2])
    #                 ws = separator / np.sqrt(separator.dot(separator))
    #                 # projected point
    #                 projection = np.append(neg[:2] + (np.dot(ws, (pos[:2] - neg[:2]))) * ws, [-1])

    #                 # we use projected point to find mid point
    #                 mid_point = (pos[0:2] + projection[0:2]) / 2
    #                 w = np.array(pos[:-1] - projection[:-1])
    #                 w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
    #                 b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
    #                 # plot_hyper_binary(w, b, data)
    #                 if margin <= compute_margin(training_data, w, b):
    #                     margin = compute_margin(training_data, w, b)
    #                     s_last = np.array([pos, neg, neg1])
    #                     w_last = w
    #                     b_last = b
    print('The answer is:--------------')
    plot_hyper_binary(w_last, b_last, data)
    print(w_last, b_last, s_last)
    return w_last, b_last, s_last

def perpendicular_bisector_of_two_points(A, B):
    mid_point = (A[0:2] + B[0:2]) / 2
    w = np.array(A[:-1] - B[:-1])
    w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
    b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
    return w, b
def projection_of_A_onto_line_BC(A,B,C):
    BA = A - B
    BC = C - B
    H = B + ((np.dot(BA, BC)) / (np.dot(BC, BC))) * BC
    return H

def distance_point_to_hyperplane(pt, w, b):
    return np.abs(((pt[0] * w[0]) + (pt[1] * w[1]) + b) / (np.sqrt((w[0] * w[0]) + (w[1] * w[1]))))

def compute_margin(data, w, b):
    margin = distance_point_to_hyperplane(data[0, :-1], w, b)
    print('first margin_of_compute_margin', margin)

    for pt in data:
        print('loop time .............')
        distance = distance_point_to_hyperplane(pt[:-1], w, b)
        print('distance of point', distance)
        if distance < margin:
            margin = distance_point_to_hyperplane(pt[:-1], w, b)
            print('update margin', margin)
        if svm_test_brute(w, b, pt) != pt[2]:
            return 0
    return margin

def svm_test_brute(w, b, x):
    if np.dot(w, x[:-1]) + b > 0:
        return 1
    else:
        return -1


# a function generates binary training data, case: num=1

def generate_training_data_binaray():
    data = np.zeros((10,3))
    for i in range(5):
        data[i] = [i-5, 0, 1]
        data[i+5] = [i+1, 0, -1]
    return data
def generate_training_data_binary(num=2):
    data = np.zeros((10,3))
    for i in range(5):
        data[i] = [0, i-5, 1]
        data[i+5] = [0, i+1, -1]
    return data
def plot_training_data_binary(data):
    for item in data:
        if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+')
        else:
            plt.plot(item[0], item[1], 'ro')
    plt.show()

def plot_hyper_binary(w, b, data):
    line = np.linspace(-10, 10)
    if w[1] != 0:
        plt.plot(line, (-w[0] * line -b) / w[1])
    else:
        plt.axvline(x=-b)
    plot_training_data_binary(data)

# w, b = perpendicular_bisector_of_two_points(pos, neg)
def perpendicular_bisector_of_two_points(A, B):
    mid_point = (A[0:2] + B[0:2]) / 2
    w = np.array(A[:-1] - B[:-1])
    w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
    b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
    return w, b

svm_train_brute(data)
