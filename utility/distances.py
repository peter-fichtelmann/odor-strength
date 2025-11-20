import numpy as np

class Distances:
    def cosine_distance(self, v1, v2):
        if np.any(v1 == 0) or np.any(v2 == 0):
            print('Warning: Zero value(s) in embedding.')
        if len(v1.shape) == 1 and len(v2.shape) == 1:
            sim = np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 10e-10)
        else:
            sim = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 10e-10)
        return 1 - sim

    def euclidean_distance(self, v1, v2):
        if np.any(v1 == 0) or np.any(v2 == 0):
            print('Warning: Zero value(s) in embedding.')
        if len(v1.shape) == 1 and len(v2.shape) == 1:
            return np.sqrt(np.sum(np.square(v1 - v2)))
        return np.sqrt(np.sum(np.square(v1 - v2), axis=1))

    def manhattan_distance(self, v1, v2):
        if np.any(v1 == 0) or np.any(v2 == 0):
            print('Warning: Zero value(s) in embedding.')
        if len(v1.shape) == 1 and len(v2.shape) == 1:
            return np.sum(np.abs(v1 - v2))
        return np.sum(np.abs(v1 - v2), axis=1)