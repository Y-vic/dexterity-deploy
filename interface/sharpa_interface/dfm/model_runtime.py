"""One model worker retains the CUDA context across load, warmup and requests."""

from __future__ import annotations


class SerializedPolicy:
    def __init__(self, policy, executor):
        self.policy = policy
        self.executor = executor
        self._capabilities = executor.submit(lambda: policy.capabilities).result()

    @property
    def capabilities(self):
        return self._capabilities

    def set_seed(self, seed):
        return self.executor.submit(self.policy.set_seed, seed).result()

    def reset(self, instruction, *, demonstrations=None):
        return self.executor.submit(self.policy.reset, instruction, demonstrations=demonstrations).result()

    def predict(self, observation):
        return self.executor.submit(self.policy.predict, observation).result()

    def close(self):
        close = getattr(self.policy, "close", None)
        if close is not None:
            return self.executor.submit(close).result()
