# -- coding: utf-8 --
import os
import math
import time
import psutil
import itertools
import numpy as np
from scipy.io import loadmat
from itertools import product
from numpy.linalg import pinv
from sklearn.cluster import KMeans
from tensorly.tenalg import mode_dot
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


def read_data(path):
    params = loadmat(path, mat_dtype=True)

    D = int(params['D'][0][0])

    miu_all = params['miu_all']

    miu0 = miu_all[0][0][0]
    miu1 = miu_all[0][1][0]
    miu2 = miu_all[0][2][0]

    miu_all = [miu0, miu1, miu2]

    sigma_all = params['sigma_all']

    sigma0 = sigma_all[0][0]
    sigma1 = sigma_all[0][1]
    sigma2 = sigma_all[0][2]

    sigma_all = [sigma0, sigma1, sigma2]

    w_all = params['w_all']

    w0 = w_all[0][0]
    w1 = w_all[0][1]
    w2 = w_all[0][2]

    w_all = [w0, w1, w2]

    return D, miu_all, sigma_all, w_all


def mvnpdfNpeak(w_all, rv_all, x):
    w0, w1, w2 = w_all
    rv0, rv1, rv2 = rv_all

    p0 = rv0.pdf(x)
    p1 = rv1.pdf(x)
    p2 = rv2.pdf(x)

    p = w0 * p0 + w1 * p1 + w2 * p2

    return p


def Rotation_estimation(samples):
    samples_num = samples.shape[0]
    m = np.mean(samples, axis=0)
    M = np.tile(m, (samples_num, 1))
    samples_mean0 = (samples - M).T
    samples_Cov = (samples_mean0 @ samples_mean0.T) / samples_num
    EValue, EVector = np.linalg.eig(samples_Cov)

    for i in range(EVector.shape[1]):
        if EVector[0, i] < 0:
            EVector[:, i] = -EVector[:, i]

    idx = np.argsort(EVector[0, :])
    EVector = EVector[:, idx]

    return EVector


def tenmat(tensor, mode):
    permuted_dims = [mode] + [i for i in range(tensor.ndim) if i != mode]
    unfolded = tensor.transpose(permuted_dims).reshape((tensor.shape[mode], -1), order="F")

    return unfolded.astype(np.float64)


def Mutlimodal_density_separation(fun, xMin, xMax, N, RN, des1, des2):
    time_1 = time.time()

    D = xMin.shape[0]
    Xs = []
    RXs = []

    for d in range(D):
        Xd = np.linspace(xMin[d], xMax[d], N[d])
        Xs.append(Xd)
        r_ind = np.floor(np.linspace(0, N[d] - 1, RN[d] + 2)).astype(int)
        r_ind = r_ind[1:-1]
        r_ind = np.sort(r_ind)
        RXs.append(Xd[r_ind])

    grid_points0 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], Xs[9])))
    grid_points1 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], Xs[8], RXs[9])))
    grid_points2 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], Xs[7], RXs[8], RXs[9])))
    grid_points3 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], Xs[6], RXs[7], RXs[8], RXs[9])))
    grid_points4 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], Xs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points5 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], Xs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points6 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], Xs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points7 = np.array(list(itertools.product(RXs[0], RXs[1], Xs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points8 = np.array(list(itertools.product(RXs[0], Xs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points9 = np.array(list(itertools.product(Xs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))

    grid_points_all = np.vstack((grid_points0, grid_points1, grid_points2, grid_points3, grid_points4, grid_points5, grid_points6, grid_points7, grid_points8, grid_points9))
    grid_points_num = grid_points_all.shape[0]

    max_p = 0
    min_p = 0

    X = []

    for idx in range(grid_points_num):
        x = grid_points_all[idx, :]
        p = fun(x)

        if p > max_p:
            max_p = p

        if p < min_p:
            min_p = p

        if p > des1:
            X.append(x)

    X = np.array(X)
    Rot = Rotation_estimation(X)

    temp = 1e-10 * np.ones(Rot.shape)
    X_rot = (np.linalg.inv(Rot + temp) @ X.T).T

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X_rot)
    idx = kmeans.labels_

    cluster_rot1 = {}
    cluster_centre_rot1 = {}
    cluster_centre1 = {}

    cluster_rot = {}
    cluster_centre_rot = {}
    cluster_centre = {}

    for k in range(3):
        cluster_rot1[k] = X_rot[idx == k]
        cluster_centre_rot1[k] = np.mean(cluster_rot1[k], axis=0)
        cluster_centre1[k] = (Rot @ cluster_centre_rot1[k].T).T

        if cluster_centre1[k][0] > 0 and cluster_centre1[k][1] > 0:
            cluster_rot[1] = cluster_rot1[k]
            cluster_centre_rot[1] = cluster_centre_rot1[k]
            cluster_centre[1] = cluster_centre1[k]
        elif cluster_centre1[k][0] < 0 and cluster_centre1[k][1] > 0:
            cluster_rot[2] = cluster_rot1[k]
            cluster_centre_rot[2] = cluster_centre_rot1[k]
            cluster_centre[2] = cluster_centre1[k]
        else:
            cluster_rot[3] = cluster_rot1[k]
            cluster_centre_rot[3] = cluster_centre_rot1[k]
            cluster_centre[3] = cluster_centre1[k]

    xMin1 = [None] * 3
    xMax1 = [None] * 3

    for k in range(3):
        xMin1[k], xMax1[k] = Effective_local_region(fun, Rot, cluster_centre_rot[k + 1], des2, xMin, xMax)

    time_2 = time.time()

    total_time = time_2 - time_1

    return Rot, xMin1, xMax1, cluster_centre, cluster_centre_rot, total_time


def Effective_local_region(fun, R, centre_rot, des, xMin, xMax):
    D = len(centre_rot)
    N = 100

    endpoints = [[xMin[d], xMax[d]] for d in range(D)]

    X_list = []
    for combo in product(range(2), repeat=D):
        x = np.array([endpoints[d][combo[d]] for d in range(D)])
        x_trans = np.linalg.inv(R).dot(x)
        X_list.append(x_trans)
    X_all = np.array(X_list)

    xMin_pca = np.min(X_all, axis=0)
    xMax_pca = np.max(X_all, axis=0)

    gap = (xMax_pca - xMin_pca) / N

    xMin1 = np.zeros(D)
    xMax1 = np.zeros(D)

    for d in range(D):
        x = np.copy(centre_rot)
        N1 = int(np.ceil((centre_rot[d] - xMin_pca[d]) / gap[d]))
        for i in range(N1):
            x[d] -= gap[d]
            xx = R.dot(x)
            p = fun(xx)
            if p <= des:
                xMin1[d] = x[d]
                break
            xMin1[d] = x[d]

        x = np.copy(centre_rot)
        N2 = int(np.ceil((xMax_pca[d] - centre_rot[d]) / gap[d]))
        for i in range(N2):
            x[d] += gap[d]
            xx = R.dot(x)
            p = fun(xx)
            if p <= des:
                xMax1[d] = x[d]
                break
            xMax1[d] = x[d]

    return xMin1, xMax1


def Local_region(fun, Rot, xMin, xMax, N, RN, des1, des2):
    time_1 = time.time()

    D = xMin.shape[0]
    Xs = []
    RXs = []

    for d in range(D):
        Xd = np.linspace(xMin[d], xMax[d], N[d])
        Xs.append(Xd)
        r_ind = np.floor(np.linspace(0, N[d] - 1, RN[d] + 2)).astype(int)
        r_ind = r_ind[1:-1]
        r_ind = np.sort(r_ind)
        RXs.append(Xd[r_ind])

    grid_points0 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], Xs[9])))
    grid_points1 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], Xs[8], RXs[9])))
    grid_points2 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], Xs[7], RXs[8], RXs[9])))
    grid_points3 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], Xs[6], RXs[7], RXs[8], RXs[9])))
    grid_points4 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], Xs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points5 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], Xs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points6 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], Xs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points7 = np.array(list(itertools.product(RXs[0], RXs[1], Xs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points8 = np.array(list(itertools.product(RXs[0], Xs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points9 = np.array(list(itertools.product(Xs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))

    grid_points_all = np.vstack((grid_points0, grid_points1, grid_points2, grid_points3, grid_points4, grid_points5, grid_points6, grid_points7, grid_points8, grid_points9))
    grid_points_num = grid_points_all.shape[0]

    max_p = 0
    min_p = 0

    X = []

    for idx in range(grid_points_num):
        x = grid_points_all[idx, :]
        xx = Rot.dot(x)

        p = fun(xx)

        if p > max_p:
            max_p = p

        if p < min_p:
            min_p = p

        if p > des1:
            X.append(x)

    X = np.array(X)
    R = Rotation_estimation(X)

    Rot2 = Rot @ R

    center = np.mean(X, axis=0)

    temp = 1e-10 * np.ones(Rot.shape)
    center_rot = (np.linalg.inv(R + temp) @ center.T).T

    xMin1, xMax1 = Effective_PCA_region(fun, R, Rot2, center_rot, des2, xMin, xMax)

    time_2 = time.time()

    total_time = time_2 - time_1

    return xMin1, xMax1, Rot2, total_time


def Effective_PCA_region(fun, R, Rot2, centre_rot, des, xMin, xMax):
    D = centre_rot.shape[0]
    N = 100

    endpoints = [[xMin[d], xMax[d]] for d in range(D)]

    X_list = []
    for combo in product(range(2), repeat=D):
        x = np.array([endpoints[d][combo[d]] for d in range(D)])
        x_trans = np.linalg.inv(R).dot(x)
        X_list.append(x_trans)
    X_all = np.array(X_list)

    xMin_pca = np.min(X_all, axis=0)
    xMax_pca = np.max(X_all, axis=0)

    gap = (xMax_pca - xMin_pca) / N

    xMin1 = np.zeros(D)
    xMax1 = np.zeros(D)

    for d in range(D):
        x = np.copy(centre_rot)
        N1 = int(np.ceil((centre_rot[d] - xMin_pca[d]) / gap[d]))
        for i in range(N1):
            x[d] -= gap[d]
            xx = Rot2.dot(x)
            p = fun(xx)
            if p <= des:
                xMin1[d] = x[d]
                break
            xMin1[d] = x[d]

        x = np.copy(centre_rot)
        N2 = int(np.ceil((xMax_pca[d] - centre_rot[d]) / gap[d]))
        for i in range(N2):
            x[d] += gap[d]
            xx = Rot2.dot(x)
            p = fun(xx)
            if p <= des:
                xMax1[d] = x[d]
                break
            xMax1[d] = x[d]

    return xMin1, xMax1


def Reconstruct_lowrank_proposal(fun, Rot, xMin, xMax, N, RN):
    time_1 = time.time()

    D = xMin.shape[0]
    Xs = []
    RXs = []
    c_ind = []

    for d in range(D):
        Xd = np.linspace(xMin[d], xMax[d], N[d])
        Xs.append(Xd)
        r_ind = np.floor(np.linspace(0, N[d] - 1, RN[d] + 2)).astype(int)
        r_ind = r_ind[1:-1]
        r_ind = np.sort(r_ind)
        c_ind.append(r_ind)
        RXs.append(Xd[r_ind])

    grid_points_0 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], Xs[9])))
    grid_points_1 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], Xs[8], RXs[9])))
    grid_points_2 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], Xs[7], RXs[8], RXs[9])))
    grid_points_3 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], Xs[6], RXs[7], RXs[8], RXs[9])))
    grid_points_4 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], Xs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points_5 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], Xs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points_6 = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], Xs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points_7 = np.array(list(itertools.product(RXs[0], RXs[1], Xs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points_8 = np.array(list(itertools.product(RXs[0], Xs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points_9 = np.array(list(itertools.product(Xs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))
    grid_points_R = np.array(list(itertools.product(RXs[0], RXs[1], RXs[2], RXs[3], RXs[4], RXs[5], RXs[6], RXs[7], RXs[8], RXs[9])))

    grid_points_all = np.vstack((grid_points_0, grid_points_1, grid_points_2, grid_points_3, grid_points_4,
                                 grid_points_5, grid_points_6, grid_points_7, grid_points_8, grid_points_9,
                                 grid_points_R))
    grid_points_num = grid_points_all.shape[0]

    C_pr = []

    for idx in range(grid_points_num):
        x = grid_points_all[idx, :]
        xx = Rot.dot(x)
        p = fun(xx)
        C_pr.append(p)

    length_list = []

    for i in (grid_points_0, grid_points_1, grid_points_2, grid_points_3, grid_points_4, grid_points_5, grid_points_6,
              grid_points_7, grid_points_8, grid_points_9, grid_points_R):
        length_list.append(int(i.size / 10))

    result = []

    current_index = 0
    for length in length_list:
        sublist = C_pr[current_index: current_index + length]
        result.append(np.array(sublist))
        current_index += length

    C_pr = result

    C_pr[0] = C_pr[0].reshape(3, 3, 3, 3, 3, 3, 3, 3, 3, 15)
    C_pr[1] = C_pr[1].reshape(3, 3, 3, 3, 3, 3, 3, 3, 15, 3)
    C_pr[2] = C_pr[2].reshape(3, 3, 3, 3, 3, 3, 3, 15, 3, 3)
    C_pr[3] = C_pr[3].reshape(3, 3, 3, 3, 3, 3, 15, 3, 3, 3)
    C_pr[4] = C_pr[4].reshape(3, 3, 3, 3, 3, 15, 3, 3, 3, 3)
    C_pr[5] = C_pr[5].reshape(3, 3, 3, 3, 15, 3, 3, 3, 3, 3)
    C_pr[6] = C_pr[6].reshape(3, 3, 3, 15, 3, 3, 3, 3, 3, 3)
    C_pr[7] = C_pr[7].reshape(3, 3, 15, 3, 3, 3, 3, 3, 3, 3)
    C_pr[8] = C_pr[8].reshape(3, 15, 3, 3, 3, 3, 3, 3, 3, 3)
    C_pr[9] = C_pr[9].reshape(15, 3, 3, 3, 3, 3, 3, 3, 3, 3)
    R = C_pr[10].reshape(3, 3, 3, 3, 3, 3, 3, 3, 3, 3)

    C_pr = C_pr[:10]

    time_2 = time.time()

    total_time = time_2 - time_1

    return C_pr[::-1], R, c_ind, total_time


def Sampling_in_interpreted_space(init_x, C, U, R, fun, xMin, xMax, Rot, N, samples_num, inter_N):
    time_1 = time.time()

    D = xMin.shape[0]
    N1 = N.copy()
    N_inter = N1 * (inter_N + 1) - inter_N

    x = init_x

    Xs = [None] * D
    gaps = np.zeros(D)
    ssig = np.zeros(D)
    Xs_min = [None] * D

    Xs_inter = [None] * D
    gaps_inter = np.zeros(D)
    ssig_inter = np.zeros(D)
    Xs_min_inter = [None] * D

    for dd in range(D):
        Xs[dd] = np.linspace(xMin[dd], xMax[dd], N1[dd])
        gaps[dd] = Xs[dd][1] - Xs[dd][0]
        ssig[dd] = (gaps[dd] / 2) ** 2
        Xs_min[dd] = Xs[dd][0] - gaps[dd] / 2

        Xs_inter[dd] = np.linspace(xMin[dd], xMax[dd], N_inter[dd])
        gaps_inter[dd] = Xs_inter[dd][1] - Xs_inter[dd][0]
        ssig_inter[dd] = (gaps_inter[dd] / 2) ** 2
        Xs_min_inter[dd] = Xs_inter[dd][0] - gaps_inter[dd] / 2

    x1 = x.copy()

    ind_inter1 = np.zeros(D, dtype=int)
    ind1 = np.zeros(D, dtype=int)
    gapss = gaps.copy()

    Xs_min_interr = Xs_min_inter
    Xs_minn = Xs_min
    Xss = Xs
    Xs_interr = Xs_inter
    gaps_interr = gaps_inter.copy()
    N_interr = N_inter.copy()

    for dd in range(D):
        x1[dd] = x[dd] + np.random.rand() * gapss[dd]
        temp = int(np.ceil((x[dd] - Xs_min_interr[dd]) / gaps_interr[dd])) - 1
        ind_inter1[dd] = np.clip(temp, 0, N_interr[dd] - 1)
        temp2 = int(np.ceil((x[dd] - Xs_minn[dd]) / gapss[dd])) - 1
        ind1[dd] = np.clip(temp2, 0, N[dd] - 1)

    xo = Rot @ x1
    p = fun(xo)

    samples_alphas = np.zeros(samples_num)
    samples_p = np.zeros(samples_num)
    samples = np.zeros((samples_num, D))
    samples_rot = np.zeros((samples_num, D))
    samples_ind_inter = np.zeros((samples_num, D), dtype=int)
    samples_ind = np.zeros((samples_num, D), dtype=int)

    Pro1 = 0

    for s in range(samples_num):

        d = (s % D)

        for dd in range(D):
            if dd != d:
                C_ind1 = C[dd][ind1[dd], :]
            else:
                C_ind1 = C[dd]

            W1 = C_ind1 @ U[dd]

            if W1.shape == (3,):
                W1 = W1.reshape((1, 3))

            if dd == 0:
                Pro1 = mode_dot(R, W1, dd)
            else:
                Pro1 = mode_dot(Pro1, W1, dd)

        P_est = tenmat(Pro1, d)

        interp_func = interp1d(Xss[d], P_est[:, 0], kind='cubic', fill_value="extrapolate")
        PDF_est = interp_func(Xs_interr[d])

        PDF_est_norm = PDF_est / np.sum(PDF_est)

        r = np.random.rand()
        s_sum = 0
        i_sel = 0
        for i in range(N_interr[d]):
            s_sum += PDF_est_norm[i]
            if r < s_sum:
                i_sel = i
                break

        ind_inter_new1 = ind_inter1.copy()
        ind_inter_new1[d] = i_sel

        x_new = x1.copy()
        x_new[d] = Xs_interr[d][i_sel] + (np.random.rand() - 0.5) * gaps_interr[d]
        ind_new1 = ind1.copy()
        temp_new = int(np.ceil((x_new[d] - Xs_minn[d]) / gapss[d])) - 1
        ind_new1[d] = np.clip(temp_new, 0, N[d] - 1)

        xx_new = Rot.dot(x_new)
        xo_new = xx_new
        p_new = fun(xx_new)

        q_new = PDF_est_norm[ind_inter_new1[d]]
        q = PDF_est_norm[ind_inter1[d]]
        alpha = min(1, p_new * q / (p * q_new))
        if np.random.rand() < alpha:
            x1[d] = x_new[d]
            xo = xo_new
            ind_inter1[d] = ind_inter_new1[d]
            ind1[d] = ind_new1[d]
            p = p_new

        samples_alphas[s] = alpha
        samples_p[s] = p
        samples[s, :] = xo
        samples_rot[s, :] = x1
        samples_ind_inter[s, :] = ind_inter1
        samples_ind[s, :] = ind1

    samples_inf = [samples, samples_p, samples_alphas, samples_rot]

    time_2 = time.time()

    total_time = time_2 - time_1

    return samples_inf, total_time


def Global_alignment(all_samples, all_P, samples_num, samples_num2, w_lims):
    time_1 = time.time()

    NW = 50
    n_chains = 3

    samples_miu_chains = []
    samples_sigma_chains = []
    samples_w_chains = []

    for i in range(n_chains):
        gm = GaussianMixture(n_components=2, max_iter=1000)
        gm.fit(all_samples[i])
        samples_miu_chains.append(gm.means_)
        samples_sigma_chains.append(gm.covariances_)
        samples_w_chains.append(gm.weights_)

    all_Q = []

    for i in range(n_chains):
        Q = np.zeros(all_samples[i].shape[0])
        for k in range(len(samples_w_chains[i])):
            pdf_vals = multivariate_normal.pdf(all_samples[i],
                                               mean=samples_miu_chains[i][k],
                                               cov=samples_sigma_chains[i][k])
            Q += samples_w_chains[i][k] * pdf_vals
        all_Q.append(Q)

    wMin1, wMax1 = w_lims[0][0], w_lims[0][1]
    wMin2, wMax2 = w_lims[1][0], w_lims[1][1]
    W1 = np.linspace(wMin1, wMax1, NW)
    W2 = np.linspace(wMin2, wMax2, NW)

    mean_alphas = np.zeros((NW, NW))
    best_w_temp = np.zeros(3)

    for i in range(NW):
        best_w_temp[0] = W1[i]
        for j in range(NW):
            best_w_temp[1] = W2[j]
            best_w_temp[2] = 1 - best_w_temp[0] - best_w_temp[1]
            if best_w_temp[0] + best_w_temp[1] >= 1:
                continue
            _, mean_alpha_val, _, _ = Generation_of_Gobal_chain(all_samples, all_P, all_Q, samples_num, best_w_temp)
            mean_alphas[i, j] = mean_alpha_val

    mean_alphas_1d = np.reshape(mean_alphas.T, (1, -1), order='F').flatten()

    max_mean_alpha = np.max(mean_alphas_1d)
    ind = np.argmax(mean_alphas_1d) + 1

    ind2 = ind % NW
    if ind2 == 0:
        ind2 = NW

    ind1 = math.ceil(ind / NW)

    best_w = np.zeros(3)
    best_w[0] = W1[ind1 - 1]
    best_w[1] = W2[ind2 - 1]
    best_w[2] = 1 - best_w[0] - best_w[1]

    samples, mean_alpha, samples_alphas, samples_p = Generation_of_Gobal_chain(all_samples, all_P, all_Q, samples_num2,
                                                                               best_w)

    time_2 = time.time()

    total_time = time_2 - time_1

    return best_w, max_mean_alpha, W1, W2, mean_alphas, samples, mean_alpha, samples_alphas, samples_p, total_time


def Generation_of_Gobal_chain(all_samples, all_P, all_Q, samples_num, ws):
    N = [s.shape[0] for s in all_samples]
    D = all_samples[0].shape[1]

    samples = np.zeros((samples_num, D))
    samples_alphas = np.zeros(samples_num)
    samples_p = np.zeros(samples_num)

    g_ind = 0
    s_ind = np.random.randint(0, N[g_ind])
    x = all_samples[g_ind][s_ind, :]
    p = all_P[g_ind][s_ind]
    q = all_Q[g_ind][s_ind] * ws[g_ind]

    for s in range(samples_num):
        r = np.random.rand()
        if r < ws[0]:
            g_ind_new = 0
        elif r < (ws[0] + ws[1]):
            g_ind_new = 1
        else:
            g_ind_new = 2

        s_ind_new = np.random.randint(0, N[g_ind_new])
        x_new = all_samples[g_ind_new][s_ind_new, :]
        p_new = all_P[g_ind_new][s_ind_new]
        q_new = all_Q[g_ind_new][s_ind_new] * ws[g_ind_new]

        if p * q_new == 0:
            alpha = 1.0
        else:
            alpha = min(1, (p_new * q) / (p * q_new))

        r = np.random.rand()

        if r < alpha:
            p = p_new
            q = q_new
            x = x_new

        samples_alphas[s] = alpha
        samples[s, :] = x
        samples_p[s] = p

    mean_alpha = np.mean(samples_alphas)

    return samples, mean_alpha, samples_alphas, samples_p
