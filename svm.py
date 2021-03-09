
import numpy as np
import matplotlib.pyplot as plt 
# one positive - two negative sv
data = np.array([[2, 1, -1], [-4,0,-1], [-3, 5, 1], [3,-5,-1]])

def svm_train_brute(training_data):
    positive = training_data[training_data[:, 2] == 1]
    negative = training_data[training_data[:, 2] == -1]
    margin = -9999999
    s_last, w_last, b_last = None, None, None
    for neg in negative:
        for neg1 in negative:
            for pos in positive:
                if neg[0] != neg1[0] and neg[1] != neg1[1]:
                    projection = projection_of_A_onto_line_BC(neg1, neg, pos,i=2)
                    w, b = perpendicular_bisector_of_two_points(pos, projection)
                    plot_hyper_binary(w, b, data)
                    if margin <= compute_margin(training_data, w, b):
                        margin = compute_margin(training_data, w, b)
                        s_last = np.array([pos, neg, neg1])
                        w_last = w
                        b_last = b
    print('The answer is:--------------')
    plot_hyper_binary(w_last, b_last, data)
    return w_last, b_last, s_last

def perpendicular_bisector_of_two_points(A, B):
    mid_point = (A[0:2] + B[0:2]) / 2
    w = np.array(A[:-1] - B[:-1])
    w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
    b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
    return w, b

def projection_of_A_onto_line_BC(C, B, A,i):
    BA = (A - B)[:i]
    BC = (C - B)[:i]
    H = B[:i] + ((np.dot(BA, BC)) / (np.dot(BC, BC))) * BC
    H = np.append(H, A[-1])
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

svm_train_brute(data)
