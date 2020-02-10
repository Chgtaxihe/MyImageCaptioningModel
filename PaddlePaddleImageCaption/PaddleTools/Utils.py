import numpy as np

def read_file(path, mode='r'):
    with open(path, mode=mode, encoding='utf-8') as f:
        context = f.read()
    return context

def write_file(path, context, mode='w'):
    with open(path, mode=mode, encoding='utf-8') as f:
        f.write(context)

def print_logger(x):
    print(x)

def orthogonal(shape, gain=1.0):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are "
                            "supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return floatX(gain * q)
