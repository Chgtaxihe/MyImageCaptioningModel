import numpy as np
import os
import os.path as path
from PIL import Image
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.framework import Parameter

IMAGE_PATH = r'/home/aistudio/flickr30k'
EXPORT_PATH = r'/home/aistudio/flickr30k_feature'
PRETRAIN_PATH = r'/home/aistudio/MobileNetV2_pretrained'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_mean = np.array(mean).reshape((3, 1, 1)).astype(np.float32)
img_std = np.array(std).reshape((3, 1, 1)).astype(np.float32)

def vgg(image):
    from model import vgg
    model = vgg.VGG19()
    return model.net(image)
    
def mobileNet(image):
    from model.MobileNetV2 import MobileNetV2
    model = MobileNetV2(trainable=False)
    return model.net(image)

def _read_image_norm(path, shape=[224, 224]):
    image = Image.open(path)
    image = image.resize(shape,Image.ANTIALIAS) if shape is not None else image
    image = np.array(image).astype(np.float32).transpose([2, 0, 1])
    image = (image / 255.) - img_mean
    image /= img_std
    
    return image

def _load_pretrain_param(executor, dirname, main_program):
    def predicate(var):
        if not isinstance(var, Parameter): return False
        file_path = os.path.normpath(os.path.join(dirname, var.name))
        if not os.path.isfile(file_path):
            print('ERROR: %s not found!'%var.name)
            return False
        return True

    fluid.io.load_vars(executor, dirname, main_program=main_program,
        predicate=predicate)

def main():
    image = layers.data('image', shape=[3, 224, 224], dtype='float16')
    feature = mobileNet(image)

    exe = fluid.Executor(fluid.CUDAPlace(0))
    exe.run(program=fluid.default_startup_program())
    _load_pretrain_param(exe, PRETRAIN_PATH, fluid.default_main_program())

    files = os.listdir(IMAGE_PATH)
    files = [f for f in files if f.endswith('jpg')]
    print("Total: %d"%len(files))
    for idx, f in enumerate(files):
        img = _read_image_norm(path.join(IMAGE_PATH, f))
        fea = exe.run(fluid.default_main_program(), feed={'image': np.expand_dims(img, 0)}, fetch_list=[feature])[0]
        np.save(path.join(EXPORT_PATH, f), fea, allow_pickle=True)
        if idx % 100 is 0:
            print("%d done." % idx)

if __name__ == "__main__":
    main()
