
def read_file(path, mode='r'):
    with open(path, mode=mode, encoding='utf-8') as f:
        context = f.read()
    return context


def write_file(path, context, mode='w'):
    with open(path, mode=mode, encoding='utf-8') as f:
        f.write(context)


def get_predicate(path, warning=True):
    """
        获取用于加载权重的predicate
    """
    def predicate(var):
        from paddle.fluid.framework import Parameter
        import os
        if not isinstance(var, Parameter): return False
        file_path = os.path.normpath(os.path.join(path, var.name))
        if not os.path.isfile(file_path) and warning:
            print('ERROR: %s not found!' % var.name)
            return False
        return True

    return predicate
