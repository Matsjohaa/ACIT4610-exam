import numpy as np

def sphere(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.dot(x, x))  # stable and fast

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))

def rastrigin(x: np.ndarray) -> float:
    n = x.size
    return float(10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

def ackley(x):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    a, b, c = 20.0, 0.2, 2*np.pi
    mean_sq = np.dot(x, x) / n
    mean_cos = np.mean(np.cos(c * x))
    # Guard against tiny negative due to FP roundoff inside sqrt:
    return float(-a * np.exp(-b * np.sqrt(max(mean_sq, 0.0)))
                 - np.exp(mean_cos) + a + np.e)

FUNCTIONS = {
    "sphere":   {"f": sphere,     "bounds": (-5.12, 5.12)},
    "rosenbrock":{"f": rosenbrock,"bounds": (-5.0, 10.0)},
    "rastrigin":{"f": rastrigin,  "bounds": (-5.12, 5.12)},
    "ackley":   {"f": ackley,     "bounds": (-32.768, 32.768)},
}

SUCCESS_THRESHOLDS = {
    "sphere": 1e-8,
    "rosenbrock": 1e-8,
    "rastrigin": 1e-4,
    "ackley": 1e-4,
}