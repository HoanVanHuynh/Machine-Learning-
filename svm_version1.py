# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import numpy as np
import matplotlib.pyplot as plt 

data = np.array([[1, 1/2, 1], [0,1,1], [3/2, 2, 1], [2,-2,1], [-2, -2, -1]])
data2 = np.array([[5, 8, 1], [8,8,1], [9, 11, -1], [1.5,1.8,-1], [1, 2, -1]])
data3 = np.array([[1, 1/2, 1], [0,1,1], [3/2, 2, 1], [2,-2,1], [-2, -2, -1]])


def svm_train_brute(training_data):

    training_data = np.asarray(training_data)

    positive = training_data[training_data[:, 2] == 1]
    negative = training_data[training_data[:, 2] == -1]

    # initial margin with negative value
    margin = -9999999
    # we will use new variables because they will update
    s_last, w_last, b_last = None, None, None

    # for pos in positive:
    #     for neg in negative:
    #         mid_point = (pos[0:2] + neg[0:2]) / 2
    #         w = np.array(pos[:-1] - neg[:-1])
    #         w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
    #         b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])

    #         # w, b = perpendicular_bisector_of_two_points(pos, neg)
    #         # def perpendicular_bisector_of_two_points(A, B):
    #         #     mid_point = (A[0:2] + B[0:2]) / 2
    #         #     w = np.array(A[:-1] - B[:-1])
    #         #     w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
    #         #     b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
    #             return w, b
    #         if margin <= compute_margin(training_data, w, b):
    #             margin = compute_margin(training_data, w, b)
    #             s_last = np.array([pos, neg])
    #             w_last = w
    #             b_last = b
    # for pos in positive:
    #     for pos1 in positive:
    #         for neg in negative:
    #             if (pos[0] != pos1[0]) and (pos[1] != pos1[1]):
    #                 separator = (pos1 - pos)[:2]
    #                 ws = separator / np.sqrt(separator.dot(separator))

    #                 projection = np.append(pos[:2] + (np.dot(ws, (neg[:2] - pos[:2]))) * ws, [1])

    #                 def projection_of_A_onto_line_BC(A,B,C,i):
    #                     A = A[:i]
    #                     B = B[:i]
    #                     C = C[:i]
    #                     BA = (A - B)
    #                     BC = C - B
    #                     H = B + ((np.dot(BA, BC)) / (np.dot(BC, BC))) * BC
    #                     H = np.append(H, A[-1])
    #                     return H
                        
    #                 # we use projected point to find mid point
    #                 mid_point = (projection[0:2] + neg[0:2]) / 2
    #                 w = np.array(projection[:-1] - neg[:-1])
    #                 # w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
    #                 w = w / np.sqrt(w.dot(w))
    #                 b = -1 * (w.dot(mid_point))

    #                 if margin <= compute_margin(training_data, w, b):
    #                     margin = compute_margin(training_data, w, b)
    #                     s_last = np.array([pos, pos1, neg])
    #                     w_last = w
    #                     b_last = b


    # one positive - two negative sv
    for neg in negative:
        for neg1 in negative:
            for pos in positive:
                if neg[0] != neg1[0] and neg[1] != neg1[1]:
                    # separator = (neg1[:2] - neg[:2])
                    # ws = separator / np.sqrt(separator.dot(separator))
                    # # projected point
                    # projection = np.append(neg[:2] + (np.dot(ws, (pos[:2] - neg[:2]))) * ws, [-1])

                    # def projection_2(neg1, neg, pos):
                    #     separator = (neg1 - neg)[:2]
                    #     ws = separator / np.sqrt(separator.dot(separator))
                    #     projection = np.append(neg[:2] + (np.dot(ws, (pos[:2] - neg[:2]))) * ws, [1])
                    #     return projection

                    projection = projection_of_A_onto_line_BC(neg1, neg, pos,i=2)

                    # def projection_of_A_onto_line_BC(C, B, A,i):
                    #     BA = (A - B)[:i]
                    #     BC = (C - B)[:i]
                    #     H = B[:i] + ((np.dot(BA, BC)) / (np.dot(BC, BC))) * BC
                    #     H = np.append(H, A[-1])
                    #     return H

                    # we use projected point to find mid point
                    # mid_point = (pos[0:2] + projection[0:2]) / 2
                    # w = np.array(pos[:-1] - projection[:-1])
                    # w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
                    # b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])

                    w, b = perpendicular_bisector_of_two_points(pos, projection)

                    # def perpendicular_bisector_of_two_points(A, B):
                    #     mid_point = (A[0:2] + B[0:2]) / 2
                    #     w = np.array(A[:-1] - B[:-1])
                    #     w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
                    #     b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
                    #     return w, b
                        
                    if margin <= compute_margin(training_data, w, b):
                        margin = compute_margin(training_data, w, b)
                        s_last = np.array([pos, neg, neg1])
                        w_last = w
                        b_last = b
    return w_last, b_last, s_last


# distance can't be negative we return absolute value
def distance_point_to_hyperplane(pt, w, b):
    return np.abs(((pt[0] * w[0]) + (pt[1] * w[1]) + b) / (np.sqrt((w[0] * w[0]) + (w[1] * w[1]))))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
def compute_margin(data, w, b):
    margin = distance_point_to_hyperplane(data[0, :-1], w, b)

    for pt in data:
        distance = distance_point_to_hyperplane(pt[:-1], w, b)
        if distance < margin:
            margin = distance_point_to_hyperplane(pt[:-1], w, b)
        if svm_test_brute(w, b, pt) != pt[2]:
            return 0

    return margin


def svm_test_brute(w, b, x):
    if np.dot(w, x[:-1]) + b > 0:
        return 1
    else:
        return -1

def projection_of_A_onto_line_BC(C, B, A,i):
    BA = (A - B)[:i]
    BC = (C - B)[:i]
    H = B[:i] + ((np.dot(BA, BC)) / (np.dot(BC, BC))) * BC
    H = np.append(H, A[-1])
    return H
    
def perpendicular_bisector_of_two_points(A, B):
    mid_point = (A[0:2] + B[0:2]) / 2
    w = np.array(A[:-1] - B[:-1])
    w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
    b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
    return w, b

# def svm_train_multiclass(training_data):
#     w2, b2 = [], []
#     training_data = np.array(training_data)
#     num_of_classes = training_data[1]

#     # we don't have class labels with 0, so we need to start from 1
#     for y in range(1, num_of_classes + 1):
#         only_data = np.copy(training_data[0])
#         for dt in only_data:
#             # one vs rest
#             if dt[2] == y:
#                 dt[2] = 1
#             else:
#                 dt[2] = -1
#         wtmp, btmp, stmp = svm_train_brute(only_data)

#         w2.append(wtmp)
#         b2.append(btmp)

#     return [w2, b2]

# training_data = np.array([])
# def svm_test_multiclass(W, B, x):
#     # initial label
#     label = -1
#     # initial distance from data point to hyperplane
#     dist_from_hyper = 0
#     for i in range(0, len(W)):
#         # we are predicting the current label as one vs rest
#         pred = svm_test_brute(W[i], B[i], x)
#         # distance to hyperplane
#         tmp_dist = np.abs(distance_point_to_hyperplane(x, W[i], B[i]))

#         # if prediction is correct
#         # and just in case we control the distances
#         if pred == 1 and tmp_dist > dist_from_hyper:
#             label = i
#             dist_from_hyper = tmp_dist

#     # iteration starts from 0 due to the array restricts, we add 1 to return correct label
#     return label + 1




# # a function generates binary training data, case: num=1

# def generate_training_data_binaray():
#     data = np.zeros((10,3))
#     for i in range(5):
#         data[i] = [i-5, 0, 1]
#         data[i+5] = [i+1, 0, -1]
#     return data
# def generate_training_data_binary(num=2):
#     data = np.zeros((10,3))
#     for i in range(5):
#         data[i] = [0, i-5, 1]
#         data[i+5] = [0, i+1, -1]
#     return data

# def generate_training_data_multi()
def plot_training_data_binary(data):
    for item in data:
        if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+')
        else:
            plt.plot(item[0], item[1], 'ro')
    plt.show()

def plot_hyper_binary(w, b, data):
    line = np.linspace(-100, 100)
    if w[1] != 0:
        plt.plot(line, (-w[0] * line -b) / w[1])
    else:
        plt.axvline(x=b)
    plot_training_data_binary(data)
