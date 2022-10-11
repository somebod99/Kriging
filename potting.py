import utils

import numpy as np
import matplotlib.pyplot as pl


def main():
    data_file_name = input("Enter file name with initial data: ")

    size_time_series = utils.enter_size_time_series()
    size_sampling = utils.enter_size_sampling(size_time_series)

    time_series = utils.get_time_series_from_file(size_time_series + size_time_series - size_sampling, data_file_name)

    '''
    list_forecast = np.zeros(size_time_series - size_sampling)

    for i in range(size_time_series - size_sampling):
        list_auto_cov_func = utils.get_list_auto_cov_func(time_series, size_time_series)

        q_ll = utils.get_q_ll(size_time_series - 1, size_time_series - 1, list_auto_cov_func)
        q_ft = utils.get_q_ft(1, size_time_series - 1, list_auto_cov_func)

        list_sampling = time_series[1:]

        forecast = np.matmul(q_ft, np.linalg.inv(q_ll)).dot(list_sampling)
        list_forecast[i] = forecast

        time_series = np.append(time_series, forecast)
        size_time_series += 1

        print(forecast)
        
    '''

    list_auto_cov_func = utils.get_list_auto_cov_func(time_series, size_time_series)

    q_ll = utils.get_q_ll(size_sampling, size_sampling, list_auto_cov_func)
    q_ft = utils.get_q_ft(size_time_series - size_sampling, size_sampling, list_auto_cov_func)

    list_sampling = time_series[size_time_series - size_sampling:size_time_series]

    forecast = np.matmul(q_ft, np.linalg.inv(q_ll)).dot(list_sampling)

    print(forecast)

    x = np.linspace(size_time_series + 1, size_time_series + forecast.size, forecast.size, dtype=int)
    y = [time_series[i + size_time_series] - forecast[i] for i in range(forecast.size)]

    pl.subplot(2, 1, 1)
    pl.plot(x, time_series[size_time_series:], x, forecast)
    pl.xlabel("N")
    pl.ylabel("Xp")
    pl.legend(["Фактические значения", "Прогноз"])
    pl.grid(True)

    pl.subplot(2, 1, 2)
    pl.plot(x, y)
    pl.xlabel("N")
    pl.ylabel("Погрешность")
    pl.grid(True)
    pl.show()


if __name__ == "__main__":
    main()
