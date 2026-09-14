import numpy as np
import torch

TOLERANCE = 1e-5
EPSILON = 1e-5


# Load Data
x = np.loadtxt("tests/data/layernorm_input.txt", dtype = np.float32)
gamma = np.loadtxt("tests/data/layernorm_gamma.txt", dtype = np.float32)
beta = np.loadtxt("tests/data/layernorm_beta.txt", dtype = np.float32)

cpp_output = np.loadtxt("tests/data/layernorm_cpp_output.txt", dtype = np.float32)



# NumPy Reference
mean = np.mean(x)
variance = np.mean((x-mean)**2)
normalized = (x-mean) / np.sqrt(variance + EPSILON)

numpy_output = gamma*normalized + beta



# PyTorch Reference
torch_input = torch.tensor(x, dtype = torch.float32)
torch_gamma = torch.tensor(gamma, dtype = torch.float32)
torch_beta = torch.tensor(beta, dtype = torch.float32)

torch_output = torch.nn.functional.layer_norm(
    torch_input, 
    normalized_shape = (x.size,),  # 对每个输入token的最后几个维度做normalization。
    weight = torch_gamma,
    bias = torch_beta,
    eps = EPSILON
).numpy()



# Error Calculation
numpy_error = np.max(np.abs(cpp_output - numpy_output))
torch_error = np.max(np.abs(cpp_output - torch_output))

print("Numpy max absolute error: ", numpy_error)
print("PyTorch ma xabsolute error: ", torch_error)


assert numpy_error < TOLERANCE, (
    f"Numpy validation failed: {numpy_error}"
)


assert torch_error < TOLERANCE, (
    f"Torch validation failed: {torch_error}"
)


print("LayerNorm correctness: PASS")