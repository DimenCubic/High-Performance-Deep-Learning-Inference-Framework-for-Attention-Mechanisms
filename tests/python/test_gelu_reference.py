import numpy as np
import torch
import math

TOLERANCE = 1e-5

# Load data
x = np.loadtxt("tests/data/gelu_input.txt", dtype = np.float32)

cpp_output = np.loadtxt("tests/data/gelu_cpp_output.txt", dtype = np.float32)



# Numpy Reference (Exact GRLU)
erf_values = np.vectorize(math.erf)(x/np.sqrt(2.0))   # Vectorize erf function, increase parallel.

numpy_output = (0.5 * x * (1.0 + erf_values)).astype(np.float32)



# PyTorch reference
torch_input = torch.tensor(x, dtype = torch.float32)

torch_output = torch.nn.functional.gelu(torch_input, approximate = "none").numpy()    # return an torch tensor, so transfer to numpy to compare.




# Error calculation
numpy_error = np.max(np.abs(cpp_output - numpy_output))
torch_error = np.max(np.abs(cpp_output - torch_output))


print( "NumPy max absolute error:", numpy_error)

print( "PyTorch max absolute error:", torch_error)


assert numpy_error < TOLERANCE, (f"Numpy Validation Failed: {numpy_error}")
assert torch_error < TOLERANCE, (f"PyTorch Validation Failed: {torch_error}")


print("GELU corrctness: PASS")

