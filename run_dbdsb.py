import torch
import hydra
import os

from bridge.trainer_dbdsb import IPF_DBDSB
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets
from accelerate import Accelerator


def run(args):
    accelerator = Accelerator(cpu=args.device == 'cpu', split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())
    accelerator.print(accelerator.state)
    print(str(torch.cuda.device_count()) + '\n\n\n\n\n\n')
    
    print(f'\n\n Config {args}\n\n')
    
    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)
    
    final_cond_model = None
    ipf = IPF_DBDSB(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                    final_cond_model=final_cond_model, valid_ds=valid_ds, test_ds=test_ds)
    accelerator.print(accelerator.state)
    # accelerator.print(ipf.net['b'])
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
    ipf.train()
