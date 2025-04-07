# -- coding: utf-8 --
from utils.utils import *

def LORA(fun, xMin, xMax, samples_num, N, RN, D):
    # ALGORITHM PROCESS:

    ### Step1. Automatic identification and separation of multiple well-separated local modes.
    Rot1, xMin1, xMax1, cluster_centre, cluster_centre_rot, time_part_1 = Mutlimodal_density_separation(fun, xMin, xMax, N, RN,
                                                                                           1e-10, 1e-10)

    ### Step2. For each local mode, estimating the rotation direction of coordinate system for more effective sampling.
    xMin2_1, xMax2_1, Rot2_1, time_part_2_1 = Local_region(fun, Rot1, xMin1[0], xMax1[0], N, RN, 1e-8,
                                                                            1e-7)

    xMin2_2, xMax2_2, Rot2_2, time_part_2_2 = Local_region(fun, Rot1, xMin1[1], xMax1[1], N, RN, 1e-8,
                                                                            1e-7)

    xMin2_3, xMax2_3, Rot2_3, time_part_2_3 = Local_region(fun, Rot1, xMin1[2], xMax1[2], N, RN, 1e-8,
                                                                            1e-7)

    time_part_2 = sum((time_part_2_1, time_part_2_2, time_part_2_3))

    # Step3. Learning the representative proposal for each local mode.
    C_pr1, R1, c_ind1, time_part_3_1 = Reconstruct_lowrank_proposal(fun, Rot2_1, xMin2_1, xMax2_1, N, RN)

    C_pr2, R2, c_ind2, time_part_3_2 = Reconstruct_lowrank_proposal(fun, Rot2_2, xMin2_2, xMax2_2, N, RN)

    C_pr3, R3, c_ind3, time_part_3_3 = Reconstruct_lowrank_proposal(fun, Rot2_3, xMin2_3, xMax2_3, N, RN)

    time_part_3 = sum((time_part_3_1, time_part_3_2, time_part_3_3))

    # Step4.Interpreting with the Gaussian kernels, and sampling with MH rule.
    time_1 = time.time()

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

    time_2 = time.time()

    samples_inf_1, time_part_4_1 = Sampling_in_interpreted_space(init_x, C, U, R, fun, xMin, xMax, Rot, N, samples_num, 2)

    time_3 = time.time()

    samples1 = samples_inf_1[0]
    samples_p1 = samples_inf_1[1]
    samples_alphas1 = samples_inf_1[2]

    time_4 = time.time()
    time_part_4_1 = (time_2 - time_1) + time_part_4_1 + (time_4 - time_3)

    time_1 = time.time()

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

    time_2 = time.time()

    samples_inf_2, time_part_4_2 = Sampling_in_interpreted_space(init_x, C, U, R, fun, xMin, xMax, Rot, N, samples_num, 2)

    time_3 = time.time()

    samples2 = samples_inf_2[0]
    samples_p2 = samples_inf_2[1]
    samples_alphas2 = samples_inf_2[2]

    time_4 = time.time()
    time_part_4_2 = (time_2 - time_1) + time_part_4_2 + (time_4 - time_3)

    time_1 = time.time()

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

    time_2 = time.time()

    samples_inf_3, time_part_4_3 = Sampling_in_interpreted_space(init_x, C, U, R, fun, xMin, xMax, Rot, N, samples_num, 2)

    time_3 = time.time()

    samples3 = samples_inf_3[0]
    samples_p3 = samples_inf_3[1]
    samples_alphas3 = samples_inf_3[2]

    time_4 = time.time()
    time_part_4_3 = (time_2 - time_1) + time_part_4_3 + (time_4 - time_3)

    samples_all = [samples1, samples2, samples3]
    samples_p_all = [samples_p1, samples_p2, samples_p3]
    samples_alphas_all = [samples_alphas1, samples_alphas2, samples_alphas3]

    time_2 = time.time()

    time_part_4 = time_part_4_1 + time_part_4_2 + time_part_4_3 + (time_2 - time_1)

    ### Step5. Global alignment of multiple learned local proposals, recovering the weighting probabilities of generated local chains and forming the global representative chain.
    best_w, max_mean_alpha, W1, W2, mean_alphas, samples, mean_alpha, samples_alphas, samples_p, time_part_5 = Global_alignment(
        samples_all, samples_p_all, 1000, 80000, [[0.1, 0.9], [0.1, 0.9]])

    peaks_inf = []
    peaks_inf.append(samples_all)
    peaks_inf.append(samples_p_all)
    peaks_inf.append(samples_alphas_all)
    peaks_inf.append(best_w)
    peaks_inf.append(max_mean_alpha)
    peaks_inf.append((W1, W2))
    peaks_inf.append(mean_alphas)

    time_list = [time_part_1, time_part_2, time_part_3, time_part_4, time_part_5]

    res = np.mean(samples_alphas)

    return samples, samples_p, samples_alphas, peaks_inf, time_list, res


def main():
    params = "./gau_params_10d3peak.mat"
    D, miu_all, sigma_all, w_all = read_data(params)

    miu0, miu1, miu2 = miu_all
    sigma0, sigma1, sigma2 = sigma_all

    rv0 = multivariate_normal(miu0, sigma0)
    rv1 = multivariate_normal(miu1, sigma1)
    rv2 = multivariate_normal(miu2, sigma2)

    rv_all = [rv0, rv1, rv2]

    def fun(x):
        return mvnpdfNpeak(w_all, rv_all, x)

    xMin = (-11) * np.ones(D, dtype=int)
    xMax = 11 * np.ones(D, dtype=int)

    N = 15 * np.ones(D, dtype=int)
    RN = 3 * np.ones(D, dtype=int)

    run_time = 3
    samples_num = 100000

    num_node = 1
    num_cpu = 1

    file_path = "log.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        psutilprocess = psutil.Process()
        psutilprocess.cpu_affinity([i for i in range(num_cpu)])

        file.write("===== ===== ===== ===== ===== num_node = %d, num_cpu = %d ===== ===== ===== ===== =====\n" % (num_node, num_cpu))

        run_time_list = [0, 0, 0, 0, 0, 0]
        run_res = 0

        for run in range(run_time):
            file.write("run " + str(run + 1) + ':\n')

            samples, samples_p, samples_alphas, peaks_inf, time_list, res = LORA(fun, xMin, xMax, samples_num, N, RN, D)

            time_list.append(sum(time_list))

            for i in range(len(time_list)):
                run_time_list[i] += time_list[i]

            run_res += res

            file.write("time_1:\t\ttime_2:\t\ttime_3:\t\ttime_4:\t\ttime_5:\t\tTotal:\n")
            for time in time_list:
                file.write("%.3fs\t\t" % time)

            file.write("\n\nres:\n" + "%.3f" % res + '\n\n')

        file.write("Average:\n")
        file.write("time_1:\t\ttime_2:\t\ttime_3:\t\ttime_4:\t\ttime_5:\t\tTotal:\n")
        for time in run_time_list:
            file.write("%.3fs\t\t" % (time / run_time))

        file.write("\n\nres:\n" + "%.3f" % (run_res / run_time) + '\n\n')

    return 0


if __name__ == "__main__":
    main()
