import numpy as np

SIZE = 1024
SEED = 42

rng = np.random.default_rng(SEED)   # Random number generator
x = rng.uniform(low = -10.0, high = 10.0, size = SIZE).astype(np.float32)

np.savetxt("tests/data/softmax_input.txt", x, fmt = "%.9f")   # Save 9 digits.


print(
    f"Generated {SIZE} random values into tests/data/softmax_input.txt"
)



