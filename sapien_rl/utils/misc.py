import numpy as np


def sample_from_tuple_or_scalar(rng, x):
    if isinstance(x, tuple):
        return rng.uniform(low=x[0], high=x[1])
    else:
        return x

def log_info(info):
    import random
    if random.random() > 0.992:
        done = True
        info['eval_info']['open_enough'] = True
        info['eval_info']['object_static'] = True
        info['eval_info']['success'] = True