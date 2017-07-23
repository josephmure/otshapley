import numpy as np

class Shapley(object):
    """Shappley indices object estimator
    """
    def __init__(self, model):
        self.model = model