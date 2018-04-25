import numpy as np


class WeightInit:

    def __init__(self, method=None, n_in=None, n_out=None):
        self.W = None

        if method == "Kaiming":
            gain = 1
            std = gain * np.sqrt(1.0 / n_in)
            a = np.sqrt(3.0) * std
            setattr(self, "w", np.random.uniform(-a, a))

        elif method == "He":
            setattr(self, "w", np.random.randn(n_in, n_out) / np.sqrt(n_in / 2))

        elif method == "Xavier":
            setattr(self, "w", np.random.normal(
                loc=0,
                scale=np.math.sqrt(2.0 / n_in),
                size=(n_in, n_out)))

    def get_weights(self):
        return getattr(self, "w")

    def init_origin(self, n_int, n_out):
        rn = np.random.RandomState(1234)
        w = rn.uniform(
            low=-np.sqrt(6. / (n_int + n_out)),
            high=np.sqrt(6. / (n_int + n_out)),
            size=(n_int, n_out)

        )
        return w

    def init_new(self, n_int, n_out):
        rn=np.random.RandomState(1234)
        w=rn.normal(
            #mean
            loc=0,
            #sd
            scale=np.math.sqrt(2.0 / n_int),
            size=(n_int, n_out)
        )
        return w