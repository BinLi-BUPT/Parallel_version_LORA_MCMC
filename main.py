# -- coding: utf-8 --
from utils import *
from LORA import LORA

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

    total_iters = 1
    samples_num = 100000

    psutilprocess = psutil.Process()
    psutilprocess.cpu_affinity([0])

    xMin = (-11) * np.ones(D, dtype=int)
    xMax = 11 * np.ones(D, dtype=int)

    N = 15 * np.ones(D, dtype=int)
    RN = 3 * np.ones(D, dtype=int)

    main_time_list = None
    main_res = 0

    # 指定文件路径
    file_path = "log.txt"

    # 使用写模式 'w'，若文件存在则会覆盖，若不存在则创建新文件
    with open(file_path, "w", encoding="utf-8") as file:
        for run in range(total_iters):
            file.write("run " + str(run + 1) + ':\n')

            print("\nrun", run + 1)

            # samples, samples_p, samples_alphas, peaks_inf, time_dict, res = LORA(fun, xMin, xMax, samples_num, N, RN, D)
            _, _, _, _, time_dict, res = LORA(fun, xMin, xMax, samples_num, N, RN, D)

            time_start, time_end_1, time_end_2, time_end_3, time_end_4, time_end_5, time_end= time_dict["time_start"], time_dict["time_end_1"], time_dict["time_end_2"], time_dict["time_end_3"], time_dict["time_end_4"], time_dict["time_end_5"], time_dict["time_end"]

            time_cost = time_end - time_start
            time_cost_1 = time_end_1 - time_start
            time_cost_2 = time_end_2 - time_end_1
            time_cost_3 = time_end_3 - time_end_2
            time_cost_4 = time_end_4 - time_end_3
            time_cost_5 = time_end_5 - time_end_4

            time_list = [time_cost, time_cost_1, time_cost_2, time_cost_3, time_cost_4, time_cost_5]

            file.write("time_cost\t\ttime_cost_1\t\ttime_cost_2\t\ttime_cost_3\t\ttime_cost_4\t\ttime_cost_5:\n")

            for time in time_list:
                file.write("%.3f\t\t" % time)

            file.write("\n\n")

            file.write("res:\n" + "%.3f" % res + '\n\n')

            if main_time_list:
                for i in range(len(main_time_list)):
                    main_time_list[i] += time_list[i]
            else:
                main_time_list = time_list

            main_res += res

        for i in range(len(main_time_list)):
            main_time_list[i] /= total_iters

        main_res /= total_iters

        file.write("\n===== ===== ===== ===== Results ===== ===== ===== =====\n")

        file.write("Average:\n")
        file.write("time_cost\t\ttime_cost_1\t\ttime_cost_2\t\ttime_cost_3\t\ttime_cost_4\t\ttime_cost_5:\n")

        for time in main_time_list:
            file.write("%.3f\t\t" % time)

        file.write("\n\n")

        file.write("res:\n" + "%.3f" % main_res + '\n\n\n')

    return 0

if __name__ == "__main__":
    main()
