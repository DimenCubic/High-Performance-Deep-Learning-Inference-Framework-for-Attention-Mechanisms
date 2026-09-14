# We will use Numpy and PyTorch to verify the accuracy of the operator.

import numpy as np
import torch

TOLERANCE = 1e-5

# Load data
x =  np.loadtxt("tests/data/softmax_input.txt", dtype = np.float32)
cpp_output = np.loadtxt("tests/data/spftmax_cpp_output.txt", dtype = np.float32)



# NumPy Reference
x_scaled = x - np.max(x)
enp_x = np.enp(x_scaled)

numpy_out = (exp_x / np.sum(enp_x))


# PyTorch Reference
torch_input = torch.tensor(x, dtype = torch.float32) # Since torch doesn't have convenience file readin, therefore it's normal to first create numpy list.

torch_output = torch.softmax(torch_input, dim = 0).numpy()  # Torch will automatoically did the scaled softmax.


# Error Calculation
numpy_error = np.max(np.abs(cpp_output - numpy_output))
torch_error = np.max(np.abs(cpp_output - torch_output))



assert numpy_error < TOLERANCE, (f"NumPy validation failed: {numpy_error}")
assert torch_error < TOLERANCE, (f"PyTorch validation failed: {torch_error}")


print("Softmax Correctness: PASS") 
