import numpy as np

SEED = 42

SEQ_LEN = 8
HIDDEN_DIM = 16
HEAD_DIM = 8

rng = np.random.default_rng(SEED)

# Input X [seq_len, hidden_dim]
X = rng.uniform(
    low = -1.0, 
    high = 1.0,
    size = (SEQ_LEN, HIDDEN_DIM)
    ).astype(np.float32)


np.savetxt("tests/data/attention_input.txt", X, fmt = "%.9f")



# Projection Weights
W_q = rng.uniform(
    low = -0.5, 
    high = 0.5,
    size = (SEQ_LEN, HIDDEN_DIM)
    ).astype(np.float32)

W_k = rng.uniform(
    low = -0.5, 
    high = 0.5,
    size = (SEQ_LEN, HIDDEN_DIM)
    ).astype(np.float32)


W_v = rng.uniform(
    low = -0.5, 
    high = 0.5,
    size = (SEQ_LEN, HIDDEN_DIM)
    ).astype(np.float32)


np.savetxt("tests/data/attention_wq.txt", W_q, fmt = "%.9f")
np.savetxt("tests/data/attention_wk.txt", W_k, fmt = "%.9f")
np.savetxt("tests/data/attention_wv.txt", W_v, fmt = "%.9f")

print("Generated attention input test data successfully")