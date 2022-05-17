import ray

@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

# Create an actor from this class.
ray.init()
counter = Counter.remote()
obj_ref = counter.increment.remote()
pass