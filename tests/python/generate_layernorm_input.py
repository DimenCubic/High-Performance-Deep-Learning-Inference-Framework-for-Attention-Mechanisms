import numpy as np

SIZE = 1024
SEED = 42

rng = np.random.default_rng(SEED)


input_data = rng.uniform(low = -10.0, high = 10.0, size = SIZE).astype(np.float32)
gamma_data = rng.uniform(low = 0.5, high = 1.5, size = SIZE).astype(np.float32)
beta_data = rng.uniform(low = -0.5, high = 0.5, size = SIZE).astype(np.float32)

np.savetxt("tests/data/layernorm_input.txt", input_data, fmt = "%.9f")
np.savetxt("tests/data/layernorm_gamma.txt", gamma_data, fmt = "%.9f")
np.savetxt("tests/data/layernorm_beta.txt", beta_data, fmt = "%.9f")


print("Generated Layernorm test data Success!\n")
print("Size: ", SIZE)