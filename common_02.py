import numpy as np


def generate_displacements(dim: int, count: int, size: float)->np.ndarray:
    directions = np.array(list(map(lambda _:  np.random.rand(dim) * 2 - 1, range(count))))
    return np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x ** 2)) * size, axis=1, arr=directions)


def displace(point: np.ndarray, displacements: np.ndarray)->np.ndarray:
    result = np.empty_like(displacements)
    for i in range(displacements.shape[0]):
        result[i, :] = displacements[i, :] + point
    return result


def generate_basis(dim: int, size: float)->np.ndarray:
    return np.concatenate((np.identity(dim), -np.identity(dim))) * size


def normalize(x: np.ndarray, size: float = 1.0) -> np.ndarray:
    length = np.sqrt(sum(x**2))
    return x if length == 0 else x * (size / length)