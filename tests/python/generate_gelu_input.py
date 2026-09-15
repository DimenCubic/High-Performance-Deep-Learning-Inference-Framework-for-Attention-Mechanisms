import numpy as np

SIZE = 1024
SEED = 42


rng = np.random.default_rng(SEED)


x = rng.uniform(low = -10.0, high = 10.0, size = SIZE).astype(np.float32)


np.savetxt("tests/data/gelu_input.txt", x, fmt = "%.9f")


print("Generated GELU test data.")
print("Size:", SIZE)