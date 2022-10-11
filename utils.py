import numpy as np


def get_auto_cov_func(time_series, m, N):
    """
    Функция вычисляет значение автоковариационной функции R(m)
    :param time_series: исходный временной ряд
    :param m: параметр автоковариационной функции R(m)
    :return: auto_cov_func / time_series.size: значение автоковариационной функции R(m)
    """

    auto_cov_func = 0.0

    for i in range(N - m):
        auto_cov_func += time_series[i + m] * (-time_series[i])

    return auto_cov_func / time_series.size


def get_list_auto_cov_func(time_series, N):
    """
    Функция вычисляет N значений автоковариационной функции R(i), i = 0,1,...,N-1
    :param time_series: исходный временной ряд
    :return: list_auto_cov_func: список значений автоковариационной функции
    """

    list_auto_cov_func = np.zeros(N)

    for i in range(N):
        list_auto_cov_func[i] = get_auto_cov_func(time_series, i, N)

    return list_auto_cov_func


def get_time_series_from_file(count_elements, file_name):
    """
    Функция считывает заданное количество элементов из файла
    :param count_elements: количество элементов, которое необходимо считать
    :param file_name: имя файла
    :return: time_series: исходный временной ряд размерности count_elements
    """

    time_series = np.zeros(count_elements, dtype=float)

    with open(file_name, "r") as file:
        for i in range(count_elements):
            time_series[i] = float(file.readline())

    return time_series


def get_q_ll(n, m, list_auto_cov_func):
    """
    Функция формирует матрицу Qll
    :param n: количество строк матрицы
    :param m: количество столбцов матрицы
    :param list_auto_cov_func: список значений автоковариационной функции
    :return: q_ll: матрица Qll
    """

    q_ll = np.zeros((n, m), dtype=float)

    for i in range(n):
        for j in range(i, m):
            q_ll[j][i] = q_ll[i][j] = list_auto_cov_func[j - i]

    return q_ll


def get_q_ft(n, m, list_auto_cov_func):
    """
    Функция формирует матрицу Qft
    :param n: количество строк матрицы
    :param m: количество столбцов матрицы
    :param list_auto_cov_func: список значений автоковариационной функции
    :return: q_ft: матрица Qft
    """

    q_ft = np.zeros((n, m), dtype=float)

    for i in range(m, n + m):
        for j in range(m):
            q_ft[i - m][j] = list_auto_cov_func[abs(i - j)]

    return q_ft


def get_prediction(time_series, start_sampling, end_sampling):
    list_auto_cov_func = get_list_auto_cov_func(time_series, time_series.size)

    size_sampling = end_sampling - start_sampling + 1
    q_ll = get_q_ll(size_sampling, size_sampling, list_auto_cov_func)
    q_ft = get_q_ft(time_series.size - size_sampling, size_sampling, list_auto_cov_func)

    list_sampling = time_series[start_sampling: end_sampling]

    return np.matmul(q_ft, np.linalg.inv(q_ll)).dot(list_sampling)


# Функции ввода
def enter_size_time_series():
    size_time_series = 0

    while size_time_series <= 0:
        size_time_series = int(input("Enter size of time series: "))

        if size_time_series <= 0:
            print("Size of time series must be bigger than 0")

    return size_time_series


def enter_size_sampling(size_time_series):
    size_sampling = 0

    while size_sampling <= 0 or size_sampling >= size_time_series:
        size_sampling = int(input("Enter size of sampling: "))

        if size_sampling <= 0 or size_sampling >= size_time_series:
            print("Size of sampling must be bigger than 0 and less than size of time series")

    return size_sampling


def enter_sampling_limits(size_sampling, size_time_series):
    start = size_time_series + 1
    end = size_time_series + 1

    while start + size_sampling - 1 > size_time_series or start == 0 or end > size_time_series:
        start = int(input("Enter index of start point for sampling: "))

        if start < 0:
            end = size_time_series + start + size_sampling
        else:
            end = start + size_sampling - 1

        if start + size_sampling - 1 > size_time_series or start == 0 or end > size_time_series:
            print(f"Incorrect index of start point")

    if start > 0:
        start -= 1

    return start, end
