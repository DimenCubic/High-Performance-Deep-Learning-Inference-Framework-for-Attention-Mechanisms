import numpy as np
import torch
import torch.nn.functional as F

TOLERANCE = 1e-5

SEQ_LEN = 8
HIDDEN_DIM = 16
HEAD_DIM = 8


# Load Input
X = np.loadtxt("tests/data/attention_input.txt", dtype = np.float32).reshape(SEQ_LEN, HIDDEN_DIM)  # Reshape to let np know we will treat it like a matrix.

W_q = np.loadtxt("tests/data/attention_wq.txt", dtype = np.float32).reshape(HIDDEN_DIM, HEAD_DIM)  
W_k = np.loadtxt("tests/data/attention_wk.txt", dtype = np.float32).reshape(HIDDEN_DIM, HEAD_DIM)  
W_v = np.loadtxt("tests/data/attention_wv.txt", dtype = np.float32).reshape(HIDDEN_DIM, HEAD_DIM)  

cpp_q = np.loadtxt("tests/data/attention_q_cpp.txt", dtype = np.float32).reshape(SEQ_LEN, HEAD_DIM)
cpp_k = np.loadtxt("tests/data/attention_k_cpp.txt", dtype = np.float32).reshape(SEQ_LEN, HEAD_DIM)
cpp_v = np.loadtxt("tests/data/attention_v_cpp.txt", dtype = np.float32).reshape(SEQ_LEN, HEAD_DIM)

cpp_scores = np.loadtxt("tests/data/attention_scores_cpp.txt", dtype = np.float32).reshape(SEQ_LEN, SEQ_LEN)
cpp_weights = np.loadtxt("tests/data/attention_weights_cpp.txt", dtype = np.float32).reshape(SEQ_LEN, SEQ_LEN)
cpp_output = np.loadtxt("tests/data/attention_output_cpp.txt", dtype = np.float32).reshape(SEQ_LEN, HEAD_DIM)



# NumPy reference
numpy_q = X @ W_q  # @ Represent matrix multiplication 
numpy_k = X @ W_k
numpy_v = X @ W_v

numpy_scores = (numpy_q @ numpy_k.T) / np.sqrt(np.float32(HEAD_DIM))

causal_mask = np.triu(np.ones((SEQ_LEN, SEQ_LEN), dtype = bool), k = 1)  # np.ones will generate a true matrix first, then triu: upper triangular will keep \ 对角线上方的元素为True
numpy_scores = np.where(causal_mask, -np.inf, numpy_scores)    

shofted_scores = numpy_scores - np.max(numpy_scores, axis = 1, keepdims = True)  # axis = 1 means find maximum value line by line, keepdims, keep the dimension, [[1],[2]] instead of [1,2]
exp_scores = np.exp(shifted_scores)
numpy_weights = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

numpy_output = numpy_weights @ numpy_v



# PyTorch Reference
torch_x = torch.tensor(X, dtype = torch.float32)
torch_wq = torch.tensor(W_q, dtype = torch.float32)
torch_wk = torch.tensor(W_k, dtype = torch.float32)
torch_wv = torch.tensor(W_v, dtype = torch.float32)

torch_q = torch_x @ torch_wq
torch_k = torch_x @ torch_wk
torch_v = torch_x @ torch_wv

torch_output = F.scaled_dot_product_attention(
    torch_q.unsqueeze(0),     # add a batch size, represent this is which set of KQV
    torch_k.unsqueeze(0),   
    torch_v.unsqueeze(0),
    is_causal = True
).squeeze(0)

# For comparing intermediate data, we will do some manual calculation here.
torch_scores = torch_q @ torch_k.transpose(0,1) / np.sqrt(HEAD_DIM)  # (0, 1) means swap dimension 0 and domension 1, standard \ diagonal swap 

torch_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype = torch.bool), diagonal = 1)

torch_scores = torch_scores.masked_fill(torch_mask, float("-inf"))

torch_weights = torch.softmax(torch_scores, dim = -1)



# Calculate scores, weights and output error.
np_finite_mask = np.isfinite(numpy_scores)   # avoid -inf - -inf will result NaN
np_scores_error = np.max(np.abs(cpp_scores[finite_mask] - numpy_scores[finite_mask]))

np_weights_error = np.max(np.abs(cpp_weights - numpy_weights))

np_output_error = np.max(np.abs(cpp_output - numpy_output))


torch_finite_mask = np.isfinite(torch_scores)   # avoid -inf - -inf will result NaN
torch_scores_error = np.max(np.abs(cpp_scores[finite_mask] - torch_scores[finite_mask]))

torch_weights_error = np.max(np.abs(cpp_weights - torch_weights))

torch_output_error = np.max(np.abs(cpp_output - torch_output))


assert np_scores_error < TOLERANCE
assert np_weights_error < TOLERANCE
assert np_output_error < TOLERANCE

assert torch_scores_error < TOLERANCE
assert torch_weights_error < TOLERANCE
assert torch_output_error < TOLERANCE


print("Attention End to End Validation success!")
