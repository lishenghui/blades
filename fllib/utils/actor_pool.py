import ray


class ActorPool(ray.util.ActorPool):
    def __init__(self, actors: list):
        super().__init__(actors)

        self._id_to_actor = {actor._ray_actor_id: actor for actor in actors}

    def get_actor_by_id(self, actor_id):
        for actor in self._actors:
            if actor._ray_actor_id == actor_id:
                return actor
        return None

    def submit(self, fn, value, affinity_actors=None):
        if affinity_actors is None:
            return super().submit(fn, value)

        for actor in affinity_actors:
            if actor in self._idle_actors:
                future = fn(actor, value)
                future_key = tuple(future) if isinstance(future, list) else future
                self._future_to_actor[future_key] = (self._next_task_index, actor)
                self._index_to_future[self._next_task_index] = future
                self._next_task_index += 1
                return

        self._pending_submits.append((fn, value, affinity_actors))
