from paddle import fluid
from model.model_base import ImageCaptionModel

if __name__ == '__main__':
    import numpy as np

    fake_image = np.zeros([1, 1280, 49], 'float32')
    fake_caption = np.zeros([1, 44], 'int64')
    mdl = ImageCaptionModel()

    x = fluid.data('image', shape=[None, 1280, 49])
    y = fluid.data('caption', shape=[None, 44], dtype='int64')
    v = mdl.training_network(x, y)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    out = exe.run(fluid.default_main_program(), feed={'image': fake_image, 'caption': fake_caption},
                  fetch_list=[v])[0]
    print(out)
