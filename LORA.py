from utils import *

def LORA(fun, xMin, xMax, samples_num, N, RN, D):
    psutilprocess = psutil.Process()
    psutilprocess.cpu_affinity([0])

    time_start = time.time()

    ### Step1. Presampling for space division and space division.
    Rot1, xMin1, xMax1, cluster_centre, cluster_centre_rot = Mutlimodal_density_separation(fun, xMin, xMax, N, RN,
                                                                                           1e-10, 1e-10)

    time_end_1 = time.time()

    ### Step2. Presampling for PCA(principal component analysis) and estimate the rotation direction using PCA in sub-space.
    max_x1, max_p1, min_x1, min_p1, xMin2_1, xMax2_1, Rot2_1 = Local_region(fun, Rot1, xMin1[0], xMax1[0], N, RN, 1e-8,
                                                                            1e-7)

    time_end_2_1 = time.time()

    max_x2, max_p2, min_x2, min_p2, xMin2_2, xMax2_2, Rot2_2 = Local_region(fun, Rot1, xMin1[1], xMax1[1], N, RN, 1e-8,
                                                                            1e-7)

    time_end_2_2 = time.time()

    max_x3, max_p3, min_x3, min_p3, xMin2_3, xMax2_3, Rot2_3 = Local_region(fun, Rot1, xMin1[2], xMax1[2], N, RN, 1e-8,
                                                                            1e-7)

    time_end_2_3 = time.time()

    time_end_2 = time.time()

    ### Step3. CUR sampling
    C_pr1, R1, c_ind1 = Reconstruct_lowrank_proposal(fun, Rot2_1, xMin2_1, xMax2_1, N, RN)

    time_end_3_1 = time.time()

    C_pr2, R2, c_ind2 = Reconstruct_lowrank_proposal(fun, Rot2_2, xMin2_2, xMax2_2, N, RN)

    time_end_3_2 = time.time()

    C_pr3, R3, c_ind3 = Reconstruct_lowrank_proposal(fun, Rot2_3, xMin2_3, xMax2_3, N, RN)

    time_end_3_3 = time.time()
    time_end_3 = time.time()

    ### mode 1
    ### 4.1 Estimate the distribution of sub-space using CUR in rotated basis.
    ss = 3 * np.ones(D, dtype=int)

    C_pr = C_pr1
    R = R1
    c_ind = c_ind1
    xMin = xMin2_1
    xMax = xMax2_1
    Rot = Rot2_1

    C = [None] * D
    U = [None] * D

    for dd in range(D):
        C[dd] = tenmat(C_pr[dd], dd)
        U[dd] = C[dd][c_ind[dd], :]
        U11, s_vals, Vh = np.linalg.svd(U[dd], full_matrices=False)

        r = ss[dd]
        S11 = np.diag(s_vals[:r])
        U11 = U11[:, :r]
        V11 = Vh[:r, :].T

        Uu = U11.T
        U[dd] = V11 @ pinv(S11) @ Uu

    init_x = xMin + (xMax - xMin) / 2

    time_end_4_1_1 = time.time()

    ### 4.2 Sampling using the PMF-gaussian kernel and MH acception guidlines.
    samples_inf_1 = Sampling_in_interpreted_space(init_x, C, U, R, fun, xMin, xMax, Rot, N, samples_num, 2)
    samples1 = samples_inf_1[0]
    samples_p1 = samples_inf_1[1]
    samples_alphas1 = samples_inf_1[2]

    mean_alpha = np.mean(samples_alphas1)
    # print(mean_alpha)

    time_end_4_1_2 = time.time()
    time_end_4_1 = time.time()

    ### mode 2
    ### 4.1 Estimate the distribution of sub-space using CUR in rotated basis.
    ss = 3 * np.ones(D, dtype=int)

    C_pr = C_pr2
    R = R2
    c_ind = c_ind2
    xMin = xMin2_2
    xMax = xMax2_2
    Rot = Rot2_2

    C = [None] * D
    U = [None] * D

    for dd in range(D):
        C[dd] = tenmat(C_pr[dd], dd)
        U[dd] = C[dd][c_ind[dd], :]
        U11, s_vals, Vh = np.linalg.svd(U[dd], full_matrices=False)

        r = ss[dd]
        S11 = np.diag(s_vals[:r])
        U11 = U11[:, :r]
        V11 = Vh[:r, :].T

        Uu = U11.T
        U[dd] = V11 @ pinv(S11) @ Uu

    init_x = xMin + (xMax - xMin) / 2

    time_end_4_2_1 = time.time()

    ### 4.2 Sampling using the PMF-gaussian kernel and MH acception guidlines.
    samples_inf_2 = Sampling_in_interpreted_space(init_x, C, U, R, fun, xMin, xMax, Rot, N, samples_num, 2)
    samples2 = samples_inf_2[0]
    samples_p2 = samples_inf_2[1]
    samples_alphas2 = samples_inf_2[2]

    mean_alpha = np.mean(samples_alphas2)
    # print(mean_alpha)

    time_end_4_2_2 = time.time()
    time_end_4_2 = time.time()

    ### mode 3
    ### 4.1 Estimate the distribution of sub-space using CUR in rotated basis.
    ss = 3 * np.ones(D, dtype=int)

    C_pr = C_pr3
    R = R3
    c_ind = c_ind3
    xMin = xMin2_3
    xMax = xMax2_3
    Rot = Rot2_3

    C = [None] * D
    U = [None] * D

    for dd in range(D):
        C[dd] = tenmat(C_pr[dd], dd)
        U[dd] = C[dd][c_ind[dd], :]
        U11, s_vals, Vh = np.linalg.svd(U[dd], full_matrices=False)

        r = ss[dd]
        S11 = np.diag(s_vals[:r])
        U11 = U11[:, :r]
        V11 = Vh[:r, :].T

        Uu = U11.T
        U[dd] = V11 @ pinv(S11) @ Uu

    init_x = xMin + (xMax - xMin) / 2

    time_end_4_3_1 = time.time()

    ### 4.2 Sampling using the PMF-gaussian kernel and MH acception guidlines.
    samples_inf_3 = Sampling_in_interpreted_space(init_x, C, U, R, fun, xMin, xMax, Rot, N, samples_num, 2)
    samples3 = samples_inf_3[0]
    samples_p3 = samples_inf_3[1]
    samples_alphas3 = samples_inf_3[2]

    mean_alpha = np.mean(samples_alphas3)
    # print(mean_alpha)

    time_end_4_3_2 = time.time()
    time_end_4_3 = time.time()

    samples_all = [samples1, samples2, samples3]
    samples_p_all = [samples_p1, samples_p2, samples_p3]
    samples_alphas_all = [samples_alphas1, samples_alphas2, samples_alphas3]

    time_end_4 = time.time()

    ### Step5. Weights estimation using gradient descent method and jump sampling in 2 chains according to estimated weights.
    best_w, max_mean_alpha, W1, W2, mean_alphas, samples, mean_alpha, samples_alphas, samples_p = Global_alignment(
        samples_all, samples_p_all, 1000, 80000, [[0.1, 0.9], [0.1, 0.9]])

    peaks_inf = {}
    peaks_inf[1] = samples_all
    peaks_inf[2] = samples_p_all
    peaks_inf[3] = samples_alphas_all
    peaks_inf[4] = best_w
    peaks_inf[5] = max_mean_alpha
    peaks_inf[6] = [W1, W2]
    peaks_inf[7] = mean_alphas

    res = np.mean(samples_alphas)

    # print("Mean of samples_alphas:", res)

    time_end_5 = time.time()
    time_end = time.time()

    time_dict = {
        "time_start": time_start,
        "time_end_1": time_end_1,
        "time_end_2_1": time_end_2_1,
        "time_end_2_2": time_end_2_2,
        "time_end_2_3": time_end_2_3,
        "time_end_2": time_end_2,
        "time_end_3_1": time_end_3_1,
        "time_end_3_2": time_end_3_2,
        "time_end_3_3": time_end_3_3,
        "time_end_3": time_end_3,
        "time_end_4_1_1": time_end_4_1_1,
        "time_end_4_1_2": time_end_4_1_2,
        "time_end_4_1": time_end_4_1,
        "time_end_4_2_1": time_end_4_2_1,
        "time_end_4_2_2": time_end_4_2_2,
        "time_end_4_2": time_end_4_2,
        "time_end_4_3_1": time_end_4_3_1,
        "time_end_4_3_2": time_end_4_3_2,
        "time_end_4_3": time_end_4_3,
        "time_end_4": time_end_4,
        "time_end_5": time_end_5,
        "time_end": time_end
    }

    return samples, samples_p, samples_alphas, peaks_inf, time_dict, res
