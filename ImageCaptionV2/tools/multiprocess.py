from multiprocessing import Queue, Process

import config


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton


def warp_function(queue_in, queue_out, func):
    while True:
        data = queue_in.get(block=True)
        if data is None:
            break
        result = func(**data)
        queue_out.put(result, block=True)

@singleton
class MultiProcessReader:

    def __init__(self):
        self.is_started = False
        self.feed_in_queue = Queue()
        self.output_queue = Queue(maxsize=config.train['data_loader_capacity'])
        self.process_list = []

    def add_request(self, **data):
        assert self.is_started
        self.feed_in_queue.put(data)

    def get_result(self):
        return self.output_queue.get(block=True)

    def start(self, max_processes, work_function):
        print('start')
        for i in range(max_processes):
            process = Process(target=warp_function,
                              args=(self.feed_in_queue, self.output_queue, work_function))
            self.process_list.append(process)
            process.daemon = True
            process.start()
        self.is_started = True

    def stop(self):
        for _ in range(len(self.process_list)):
            self.feed_in_queue.put(None)

    def stop_immediate(self):
        for process in self.process_list:
            process.terminate()

