init_from = 'scratch'  
# configs/experiment.py — 10% fineweb, quick comparison run
dataset = 'fineweb'
out_dir = 'checkpoints'

# model — best combo not yet tested
n_layer = 12
n_head = 8
n_embd = 640
dropout = 0.1
bias = False

# short run for comparison
max_iters =5000
eval_interval = 200   # checks val loss every 2000 steps
always_save_checkpoint = True  # saves whenever val loss improves


# optimizer
batch_size = 4                    # down from 12
gradient_accumulation_steps = 8   # keep effective batch size reasonable
compile = False

learning_rate = 6e-4
weight_decay = 0.1
grad_clip = 1.0

dtype = 'float16'   # safe for this GPU

warmup_iters = 500
lr_decay_iters = 5000
min_lr = 6e-5
