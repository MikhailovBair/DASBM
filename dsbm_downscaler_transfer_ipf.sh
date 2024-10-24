
exp_name="dsbm_downscaler_transfer_ipf"
path_to_save_info="experimentss"

python main.py  num_steps=30 num_iter=5000 method=dbdsb first_num_iter=20000 \
 gamma_min=0.1 gamma_max=0.1 first_coupling=ref use_minibatch_ot=False batch_size=64 \
 dataset=downscaler_transfer exp_name=${exp_name} path_to_save_info="${path_to_save_info}" start_coupling_v1="ipf"
