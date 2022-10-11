import utils

import numpy as np
import matplotlib.pyplot as pl


def main():
    # Ввод исходных данных
    data_file_name = input("Enter file name with initial data: ")

    size_time_series = utils.enter_size_time_series()
    size_sampling = utils.enter_size_sampling(size_time_series)
    start, end = utils.enter_sampling_limits(size_sampling, size_time_series)

    time_series = utils.get_time_series_from_file(size_time_series, data_file_name)

    # Вычисление значений автоковариационной функции
    list_auto_cov_func = utils.get_list_auto_cov_func(time_series, size_time_series)

    # Формирование матриц Qll и Qft
    q_ll = utils.get_q_ll(size_sampling, size_sampling, list_auto_cov_func)
    q_ft = utils.get_q_ft(size_time_series - size_sampling, size_sampling, list_auto_cov_func)

    # Выбор базового вектора
    list_sampling = time_series[start: end]

    # Расчет вектора прогноза: f = Qft * Qll^(-1) * l
    forecast = np.matmul(q_ft, np.linalg.inv(q_ll)).dot(list_sampling)

    print(forecast)

    '''
    x = np.linspace(end + 1, end + forecast.size, forecast.size, dtype=int)
    y = [time_series[i + size_sampling] - forecast[i] for i in range(forecast.size)]
    pl.plot(x, y)
    pl.grid(True)
    pl.show()
    '''

    '''
    x1 = np.linspace(1, time_series.size, time_series.size, dtype=int)
    x2 = np.linspace(end + 1, end + forecast.size, forecast.size, dtype=int)

    pl.plot(x1, time_series, x2, forecast)
    pl.xlabel("N")
    pl.legend(["Фактические значения", "Прогноз"])
    pl.grid(True)
    pl.show()
    '''

if __name__ == "__main__":
    main()
