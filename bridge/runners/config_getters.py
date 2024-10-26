import torch
from omegaconf import OmegaConf
import hydra
from ..models import *
from .plotters import *
import torchvision.datasets
import torchvision.transforms as transforms
import os
from functools import partial
from .logger import CSVLogger, WandbLogger, Logger
from torch.utils.data import DataLoader
from bridge.data.downscaler import DownscalerDataset
from .datasets import Celeba

cmp = lambda x: transforms.Compose([*x])

def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)


def get_plotter(runner, args):
    dataset_tag = getattr(args, DATASET)
    if dataset_tag in [DATASET_MNIST, DATASET_EMNIST, DATASET_CIFAR10, 'mnist_2', 'mnist_3', 'celeba_male', 'celeba_female']:
        print(f"Use ImPlotter")
        run_dir = os.path.join(args.path_to_save_info, args.data.dataset, args.exp_name)
        im_dir = os.path.join(run_dir, "im")
        gif_dir = os.path.join(run_dir, "gif")
        print(f"will save images in {im_dir}")
        print(f"will save gifs in {gif_dir}")
        return ImPlotter(runner, args, im_dir=im_dir, gif_dir=gif_dir)
    elif dataset_tag in [DATASET_DOWNSCALER_LOW, DATASET_DOWNSCALER_HIGH, DATASET_CIFAR10_LOW, DATASET_CIFAR10]:
        print(f"Use DownscalerPlotter")
        return DownscalerPlotter(runner, args)
    else:
        print(f"Use Plotter")
        return Plotter(runner, args)


# Model
# --------------------------------------------------------------------------------

MODEL = 'Model'
BASIC_MODEL = 'Basic'
UNET_MODEL = 'UNET'
DOWNSCALER_UNET_MODEL = 'DownscalerUNET'

NAPPROX = 2000


def get_model(args):
    model_tag = getattr(args, MODEL)

    if model_tag == UNET_MODEL:
        image_size = args.data.image_size

        if args.model.channel_mult is not None:
            channel_mult = args.model.channel_mult
        else:
            if image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 160:
                channel_mult = (1, 2, 2, 4)
            elif image_size == 128:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 64:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 32:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 28:
                channel_mult = (0.5, 1, 1)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
            
        attention_ds = []
        for res in args.model.attention_resolutions.split(","):
            if image_size % int(res) == 0:
                attention_ds.append(image_size // int(res))

        kwargs = {
            "in_channels": args.data.channels,
            "model_channels": args.model.num_channels,
            "out_channels": args.data.channels,
            "num_res_blocks": args.model.num_res_blocks,
            "attention_resolutions": tuple(attention_ds),
            "dropout": args.model.dropout,
            "channel_mult": channel_mult,
            "num_classes": None,
            "use_checkpoint": args.model.use_checkpoint,
            "num_heads": args.model.num_heads,
            "use_scale_shift_norm": args.model.use_scale_shift_norm,
            "resblock_updown": args.model.resblock_updown,
            "temb_scale": args.model.temb_scale
        }

        net = UNetModel(**kwargs)

    elif model_tag == DOWNSCALER_UNET_MODEL:
        image_size = args.data.image_size
        channel_mult = args.model.channel_mult

        kwargs = {
            "in_channels": args.data.channels,
            "cond_channels": args.data.cond_channels, 
            "model_channels": args.model.num_channels,
            "out_channels": args.data.channels,
            "num_res_blocks": args.model.num_res_blocks,
            "dropout": args.model.dropout,
            "channel_mult": channel_mult,
            "temb_scale": args.model.temb_scale, 
            "mean_bypass": args.model.mean_bypass,
            "scale_mean_bypass": args.model.scale_mean_bypass,
            "shift_input": args.model.shift_input,
            "shift_output": args.model.shift_output,
        }

        net = DownscalerUNetModel(**kwargs)

    return net

# Optimizer
# --------------------------------------------------------------------------------

def get_optimizer(net, args):
    lr = args.lr
    optimizer = args.optimizer
    if optimizer == 'Adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=args.weight_decay)


# Dataset
# --------------------------------------------------------------------------------

DATASET = 'Dataset'
DATASET_TRANSFER = 'Dataset_transfer'
DATASET_MNIST = 'mnist'
DATASET_EMNIST = 'emnist'
DATASET_CIFAR10 = 'cifar10'
DATASET_CIFAR10_LOW = 'cifar10_low'
DATASET_DOWNSCALER_LOW = 'downscaler_low'
DATASET_DOWNSCALER_HIGH = 'downscaler_high'
DATASET_CELEBA = 'celeba'


def get_random_colored_images(images, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    images = 0.5*(images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360*np.random.rand(size)
    
    for V, H in zip(images, hues):
        V_min = 0
        
        a = (V - V_min)*(H%60)/60
        V_inc = a
        V_dec = V - a
        
        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H/60) % 6
        
        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec
        
        colored_images.append(colored_image)
        
    colored_images = torch.stack(colored_images, dim = 0)
    colored_images = 2*colored_images - 1
    
    return colored_images


from torchvision import datasets
from torch.utils.data import TensorDataset


def load_paired_colored_mnist(target_number=2, train=True, seed=None, dataset_size=None):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 2 * x - 1)
    ])
    
    dataset = datasets.MNIST("./", train=train, transform=transform, download=True)
    
    digits = torch.stack(
            [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == target_number],
            dim=0
        )
    
    digits = digits.reshape(-1, 1, 32, 32)

    if dataset_size is not None:
        if digits.shape[0] < dataset_size:
            digits = digits.repeat([dataset_size // digits.shape[0] + 1, 1, 1, 1])[:dataset_size]

    digits_colored = get_random_colored_images(digits, seed=seed)
    
    size = digits_colored.shape[0]
    
    dataset = TensorDataset(digits_colored, torch.zeros_like(digits_colored))
    
    return dataset

def get_datasets(args):
    dataset_tag = getattr(args, DATASET)

    # INITIAL (DATA) DATASET

    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)

    print(f'\n\nFirst dataset Tag {dataset_tag}\n\n')

    # MNIST DATASET
    if dataset_tag == DATASET_MNIST:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'mnist')
        load = args.load
        assert args.data.channels == 1
        assert args.data.image_size == 28
        train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        init_ds = torchvision.datasets.MNIST(root=root, train=True, transform=cmp(train_transform), download=True)

    if dataset_tag == 'mnist_2':
        print(f"Use mnist_2!")
        init_ds = load_paired_colored_mnist(target_number=2)

    if dataset_tag == 'celeba_male':
        transform = transforms.Compose([
            transforms.Resize((args.data.image_size, args.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        init_ds = Celeba(data_dir, train=True, transform=transform, part_test=0.1, mode="male")
    
    if dataset_tag == 'celeba_female':
        transform = transforms.Compose([
            transforms.Resize((args.data.image_size, args.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        init_ds = Celeba(data_dir, train=True, transform=transform, part_test=0.1, mode="female")

    # Image Bench
        
    from eot_benchmark.image_benchmark import ImageBenchmark
    if dataset_tag == 'image_bench_target_pics':
    
        benchmark = ImageBenchmark(batch_size=512, eps=args.data.eps, glow_device=f"cuda:0",
                                samples_device=f"cuda:0", download=False)
        
        sampler_x = benchmark.X_sampler

        

    if dataset_tag == 'image_bench_noised_pics':

        benchmark = ImageBenchmark(batch_size=512, eps=args.data.eps, glow_device=f"cuda:0",
                                samples_device=f"cuda:0", download=False)
        
        sampler_y = benchmark.Y_sampler
        
    
    # CIFAR10 DATASET
    if dataset_tag == DATASET_CIFAR10:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'cifar10')
        load = args.load
        assert args.data.channels == 3
        assert args.data.image_size == 32
        train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        if args.data.random_flip:
            train_transform.insert(0, transforms.RandomHorizontalFlip())
        
        init_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=cmp(train_transform), download=True)
    
    if dataset_tag == DATASET_CIFAR10_LOW:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'cifar10')
        load = args.load
        assert args.data.channels == 3
        assert args.data.image_size == 32
        train_transform = [transforms.Resize((16, 16)), transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        if args.data.random_flip:
            train_transform.insert(0, transforms.RandomHorizontalFlip())
        
        init_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=cmp(train_transform), download=True)


    # Downscaler dataset
    if dataset_tag == DATASET_DOWNSCALER_HIGH:
        root = os.path.join(data_dir, 'downscaler')
        train_transform = [transforms.Normalize((0.,), (1.,))]
        assert not args.data.random_flip
        # if args.data.random_flip:
        #     train_transform = train_transform + [
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.RandomVerticalFlip(p=0.5),
        #         transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        #     ]
        wavenumber = args.data.get('wavenumber', 0)
        split = args.data.get('split', "train")
        
        init_ds = DownscalerDataset(root=root, resolution=512, wavenumber=wavenumber, split=split, transform=cmp(train_transform))

    # FINAL DATASET

    final_ds, mean_final, var_final = get_final_dataset(args, init_ds)
    return init_ds, final_ds, mean_final, var_final


def get_final_dataset(args, init_ds):

    if args.transfer:
        data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)
        dataset_transfer_tag = getattr(args, DATASET_TRANSFER)
        mean_final = torch.tensor(0.)
        var_final = torch.tensor(1.*10**3)  # infty like

        print(f'\n\Second dataset Tag {dataset_transfer_tag}\n\n')

        if dataset_transfer_tag == DATASET_EMNIST:
            from ..data.emnist import FiveClassEMNIST
            # data_tag = args.data.dataset
            root = os.path.join(data_dir, 'emnist')
            load = args.load
            assert args.data.channels == 1
            assert args.data.image_size == 28
            train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            final_ds = FiveClassEMNIST(root=root, train=True, download=True, transform=cmp(train_transform))

        if dataset_transfer_tag == 'mnist_3':
            final_ds = load_paired_colored_mnist(target_number=3)

        if dataset_transfer_tag == 'celeba_male':
            transform = transforms.Compose([
                transforms.Resize((args.data.image_size, args.data.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            final_ds = Celeba(data_dir, train=True, transform=transform, part_test=0.1, mode="male")

        if dataset_transfer_tag == 'celeba_female':
            transform = transforms.Compose([
                transforms.Resize((args.data.image_size, args.data.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            final_ds = Celeba(data_dir, train=True, transform=transform, part_test=0.1, mode="female")

        if dataset_transfer_tag == DATASET_CIFAR10_LOW:
            root = os.path.join(data_dir, 'cifar10')
            load = args.load
            assert args.data.channels == 3
            assert args.data.image_size == 32
            train_transform = [transforms.Resize((16, 16)), transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            if args.data.random_flip:
                train_transform.insert(0, transforms.RandomHorizontalFlip())
            
            final_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=cmp(train_transform), download=True)

        if dataset_transfer_tag == DATASET_DOWNSCALER_LOW:
            root = os.path.join(data_dir, 'downscaler')
            train_transform = [transforms.Normalize((0.,), (1.,))]
            if args.data.random_flip:
                train_transform = train_transform + [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                ]

            split = args.data.get('split', "train")
            
            final_ds = DownscalerDataset(root=root, resolution=64, split=split, transform=cmp(train_transform))

    else:
        if args.adaptive_mean:
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, num_workers=args.num_workers, worker_init_fn=worker_init_fn)))[0]
            mean_final = vec.mean(axis=0)
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
        elif args.final_adaptive:
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, num_workers=args.num_workers, worker_init_fn=worker_init_fn)))[0]
            mean_final = vec.mean(axis=0)
            var_final = vec.var(axis=0)
        else:
            mean_final = eval(args.mean_final) if isinstance(args.mean_final, str) else torch.tensor([args.mean_final])
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
        final_ds = None

    return final_ds, mean_final, var_final


def get_valid_test_datasets(args):
    valid_ds, test_ds = None, None

    dataset_tag = getattr(args, DATASET)
    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)
    
    # MNIST DATASET
    if dataset_tag == DATASET_MNIST:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'mnist')
        load = args.load
        assert args.data.channels == 1
        assert args.data.image_size == 28
        test_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        valid_ds = None
        test_ds = torchvision.datasets.MNIST(root=root, train=False, transform=cmp(test_transform), download=True)

    if dataset_tag == 'mnist_2':
        test_ds = load_paired_colored_mnist(target_number=2, train=False)

    if dataset_tag == 'celeba_male':
        transform = transforms.Compose([
            transforms.Resize((args.data.image_size, args.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_ds = Celeba(data_dir, train=False, transform=transform, part_test=0.1, mode="male")

    if dataset_tag == 'celeba_female':
        transform = transforms.Compose([
            transforms.Resize((args.data.image_size, args.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_ds = Celeba(data_dir, train=False, transform=transform, part_test=0.1, mode="female")
    
    if dataset_tag == DATASET_CIFAR10:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'cifar10')
        load = args.load
        assert args.data.channels == 3
        assert args.data.image_size == 32
        test_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        valid_ds = None
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=cmp(test_transform), download=True)

    if dataset_tag == 'cifar10_low':
        root = os.path.join(data_dir, 'cifar10')
        load = args.load
        assert args.data.channels == 3
        assert args.data.image_size == 32
        test_transform = [transforms.Resize((16, 16)), transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        valid_ds = None
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=cmp(test_transform), download=True)

    if test_ds == None:
        print(dataset_tag)
        raise 'No test dataset. Please load one!'
    
    # # CIFAR10 DATASET
    # if dataset_tag == DATASET_CIFAR10:
    #     # data_tag = args.data.dataset
    #     root = os.path.join(data_dir, 'cifar10')
    #     load = args.load
    #     assert args.data.channels == 3
    #     assert args.data.image_size == 32
    #     test_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    #     valid_ds = None
    #     test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=cmp(test_transform), download=True)

    return valid_ds, test_ds


# Logger
# --------------------------------------------------------------------------------

LOGGER = 'LOGGER'
CSV_TAG = 'CSV'
WANDB_TAG = 'Wandb'
NOLOG_TAG = 'NONE'


def get_logger(args, name):
    logger_tag = getattr(args, LOGGER)

    if logger_tag == CSV_TAG:
        kwargs = {'save_dir': args.CSV_log_dir, 'name': name, 'flush_logs_every_n_steps': 1}
        return CSVLogger(**kwargs)

    if logger_tag == WANDB_TAG:
        log_dir = os.getcwd()
        if not args.use_default_wandb_name:
            run_name = os.path.normpath(os.path.relpath(log_dir, os.path.join(
                hydra.utils.to_absolute_path(args.paths.experiments_dir_name), args.name))).replace("\\", "/")
        else:
            run_name = None
        data_tag = args.data.dataset
        config = OmegaConf.to_container(args, resolve=True)

        wandb_entity = 'NoNameWandbEntity'

        # wandb_entity = os.environ['WANDB_ENTITY']
        # assert len(wandb_entity) > 0, "WANDB_ENTITY not set"

        kwargs = {'name': run_name, 'project': 'dsbm_' + args.name, 'prefix': name, 
                  'tags': [data_tag], 'config': config, 'id': str(args.wandb_id) if args.wandb_id is not None else None}
        return WandbLogger(**kwargs)

    if logger_tag == NOLOG_TAG:
        return Logger()
