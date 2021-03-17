    
import numpy as np
def projection(pos1, pos, neg):
    separator = (pos1 - pos)[:2]
    ws = separator / np.sqrt(separator.dot(separator))
    projection = np.append(pos[:2] + (np.dot(ws, (neg[:2] - pos[:2]))) * ws, [1])
    return projection
neg = np.array([2, 5, -1])
neg1 = np.array([8, 3, -1])
pos = np.array([-3, 4, 1])
def projection_2(neg1, neg, pos):
    separator = (neg1 - neg)[:2]
    ws = separator / np.sqrt(separator.dot(separator))
    projection = np.append(neg[:2] + (np.dot(ws, (pos[:2] - neg[:2]))) * ws, [1])
    return projection
def projection_of_A_onto_line_BC(C, B, A,i):
    BA = (A - B)[:i]
    BC = (C - B)[:i]
    H = B[:i] + ((np.dot(BA, BC)) / (np.dot(BC, BC))) * BC
    H = np.append(H, A[-1])
    return H

# def projection_of_A_onto_line_BC(A,B,C):
#     BA = A - B
#     BC = C - B
#     H = B + ((np.dot(BA, BC)) / (np.dot(BC, BC))) * BC
#     return H
    
def perpendicular_bisector_of_two_points(A, B):
    mid_point = (A[0:2] + B[0:2]) / 2
    w = np.array(A[:-1] - B[:-1])
    w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
    b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
    return w, b
data = np.array([[2, 5, 1],
                [8, 3, -1]
])
A = np.array([2, 5, 1])
B = np.array([8, 3, 1])
C = np.array([-3, 4, -1])

# plot_training_data_binary(data)
# plot_hyper_binary(w, b, data)
