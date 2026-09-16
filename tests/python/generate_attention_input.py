import numpy as np

SEED = 42

SEQ_LEN = 8
HIDDEN_DIM = 16
HEAD_DIM = 8

rng = np.random.default_rng(SEED)

# Input X [seq_len, hidden_dim]
