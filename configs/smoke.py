# configs/smoke.py
dataset = 'shakespeare_char'   # → looks in data/shakespeare_char/
out_dir = 'out-smoke'
n_layer = 4
n_head = 4
n_embd = 128
max_iters = 50
eval_interval = 25
compile = False
dtype = 'float32'