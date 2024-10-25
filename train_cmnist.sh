current_T=4
current_innter_proj_iters=25000
current_proj_iters=50000
cur_ema_decay=0.999

papermill MNIST_ASBM_IPMF.ipynb -p 

#papermill MNIST_ASBM_IPMF.ipynb -p plan $2 -p T $current_T -p eps $1 -p markovian_proj_iters $current_proj_iters -p inner_ipmf_mark_proj_iters $current_innter_proj_iters -p ema_decay $cur_ema_decay  MNIST_ASBM_IPMF_log.ipynb 