class BaseSampler:
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
