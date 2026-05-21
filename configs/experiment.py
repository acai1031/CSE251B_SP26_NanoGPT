# configs/experiment.py — 10% fineweb, quick comparison run
dataset = 'fineweb'
out_dir = 'checkpoints'

# model — best combo not yet tested
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
bias = False

# short run for comparison
max_iters = 1000
eval_interval = 200
always_save_checkpoint = True
compile = False

# optimizer
learning_rate = 6e-4
weight_decay = 0.1
grad_clip = 1.0

dtype = 'float32'    # safe for this GPU