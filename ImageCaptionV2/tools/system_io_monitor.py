import os
import re
import sys
import time

pattern_io = re.compile(r'rchar: ([0-9]+)')
pattern_gpu = re.compile(r'([0-9]+%)')

def main(pid, delay):
    command_io = 'cat /proc/{}/io'.format(pid)
    command_gpu = 'nvidia-smi'
    delay = float(delay)
    prev_rchar = 0
    gpu_sum, delta_sum = 0, 0
    idx = 0
    while True:
        idx += 1
        r = os.popen(command_io)  # 执行该命令
        info = r.readlines()[0]  # 读取命令行的输出到一个list
        r.close()

        rchar = int(re.findall(pattern_io, info)[0])
        if idx != 1:
            delta = (rchar - prev_rchar) / 1024 / 1024
            delta_sum += delta
            print('Read: {:.1f} Mb/s Mean: {:.1f} Mb/s'.format(delta / delay, delta_sum / (idx - 1) / delay), end=' ')
        prev_rchar = rchar

        r = os.popen(command_gpu)  # 执行该命令
        info = r.readlines()[8]  # 读取命令行的输出到一个list
        r.close()

        used = re.findall(pattern_gpu, info)[0]
        gpu_sum += int(used[:-1])
        print('GPU: {} GPU_Mean: {:.1f}%'.format(used, gpu_sum / idx))

        time.sleep(delay)


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        print('usage: . [pid] [delay]')
    main(*args[1:])
