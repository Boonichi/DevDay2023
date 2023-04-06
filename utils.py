import logging
import time
from datetime import timedelta

def fcall(fun):
    """
    Convenience decorator used to measure the time spent while executing
    the decorated function.
    :param fun:
    :return:
    """
    def wrapper(*args, **kwargs):

        logging.info("[{}] ...".format(fun.__name__))

        start_time = time.perf_counter()
        res = fun(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        logging.info("[{}] Done! {}s\n".format(fun.__name__, timedelta(seconds=runtime)))
        return res

    return wrapper

def rmr_score(actual, predict):
    """データフレームから残差平均割合を計算する。

    Args:
        actual (np.array): 実績
        predict (np.array): 予測

    Returns:
        float: 残差平均割合
    """
    eps = 1e-9
    actual = actual + eps
    diff = actual - predict
    mx = sum(abs(diff)) / sum(actual)
    return mx * 100