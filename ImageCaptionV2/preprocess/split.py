import random
import os

root_path = r"f:\Dataset\flickr30k"
names = set()
with open(r"f:\Dataset\flickr30k\captions.token", 'r', encoding="utf-8") as f:
    for line in f.readlines():
        name = line.split("#")[0]
        names.add(name)

names = list(names)
random.shuffle(names)

# 10% 用作测试
# 5% 用作交叉验证
test_len = int(0.1 * len(names))
val_len = int(0.05 * len(names))

test_file = names[:test_len]
val_file = names[test_len: val_len + test_len]

train_file = names[test_len + val_len:]
content = [test_file, val_file, train_file]
filename = ["test.txt", "dev.txt", "train.txt"]
for idx, n in enumerate(filename):
    with open(os.path.join(root_path, n), 'w', encoding="utf-8") as f:
        for name in content[idx]:
            f.write(name + '\n')

print("测试集:{}, 验证集:{}, 训练集:{}".format(test_len, val_len, len(names) - test_len - val_len))

# 测试集:3178, 验证集:1589, 训练集:27015
# 测试集:3178, 验证集:1589, 训练集:27016
if __name__ == '__main__':
    pass
