import os

class ModelIterator:
    def __init__(self, model_dir):
        self.models = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.models):
            model_path = self.models[self.index]
            self.index += 1
            return model_path
        else:
            raise StopIteration

