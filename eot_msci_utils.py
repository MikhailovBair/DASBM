import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances
import wandb
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

from eot_benchmark.gaussian_mixture_benchmark import (
    get_guassian_mixture_benchmark_sampler,
    get_guassian_mixture_benchmark_ground_truth_sampler, 
    get_test_input_samples
)

from eot_benchmark.metrics import (
    compute_BW_UVP_by_gt_samples, compute_BW_by_gt_samples, calculate_cond_bw
)

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
    
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)
    

class TensorSampler(Sampler):
    def __init__(self, tensor, device='cuda'):
        super(TensorSampler, self).__init__(device)
        self.tensor = torch.clone(tensor).to(device)
        
    def sample(self, size=5):
        assert size <= self.tensor.shape[0]
        
        ind = torch.tensor(np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False), device=self.device)
        return torch.clone(self.tensor[ind]).detach().to(self.device)

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def pca_plot(x_0_gt, x_1_gt, x_1_pred, n_plot, save_name='plot_pca_samples.png', is_wandb=True):

    x_0_gt, x_1_gt, x_1_pred = x_0_gt.cpu(), x_1_gt.cpu(), x_1_pred.cpu()
    fig,axes = plt.subplots(1, 3,figsize=(12,4),squeeze=True,sharex=True,sharey=True)
    pca = PCA(n_components=2).fit(x_1_gt)
    
    x_0_gt_pca = pca.transform(x_0_gt[:n_plot])
    x_1_gt_pca = pca.transform(x_1_gt[:n_plot])
    x_1_pred_pca = pca.transform(x_1_pred[:n_plot])
    
    axes[0].scatter(x_0_gt_pca[:,0], x_0_gt_pca[:,1], c="g", edgecolor = 'black',
                    label = r'$x\sim P_0(x)$', s =30)
    axes[1].scatter(x_1_gt_pca[:,0], x_1_gt_pca[:,1], c="orange", edgecolor = 'black',
                    label = r'$x\sim P_1(x)$', s =30)
    axes[2].scatter(x_1_pred_pca[:,0], x_1_pred_pca[:,1], c="yellow", edgecolor = 'black',
                    label = r'$x\sim T(x)$', s =30)
    
    for i in range(3):
        axes[i].grid()
        axes[i].set_xlim([-5, 5])
        axes[i].set_ylim([-5, 5])
        axes[i].legend()
    
    fig.tight_layout(pad=0.5)
    
    im = fig2img(fig)
    im.save(save_name)
    
    if is_wandb:
        wandb.log({save_name.split('.')[0]: [wandb.Image(fig2img(fig))]})


    
def compute_condBWUVP(sample_fn, dim, eps, n_samples=1000, device='cpu'):
    test_samples = get_test_input_samples(dim=dim, device=device)
    # test_samples = test_samples[:100]
    
    model_input = test_samples.reshape(test_samples.shape[0], 1, -1).repeat(1, n_samples, 1)
    predictions = []

    with torch.no_grad():
        for test_samples_repeated in tqdm(model_input):
            predictions.append(sample_fn(test_samples_repeated).cpu())

    predictions = torch.stack(predictions, dim=0)

    # calculate cond_bw new

    new_cond_bw = calculate_cond_bw(test_samples, predictions, eps=eps, dim=dim)
    
    return new_cond_bw


def mmd(x, y):
    Kxx = pairwise_distances(x, x)
    Kyy = pairwise_distances(y, y)
    Kxy = pairwise_distances(x, y)

    m = x.shape[0]
    n = y.shape[0]
    
    c1 = 1 / ( m * (m - 1))
    A = np.sum(Kxx - np.diag(np.diagonal(Kxx)))

    # Term II
    c2 = 1 / (n * (n - 1))
    B = np.sum(Kyy - np.diag(np.diagonal(Kyy)))

    # Term III
    c3 = 1 / (m * n)
    C = np.sum(Kxy)

    # estimate MMD
    mmd_est = -0.5*c1*A - 0.5*c2*B + c3*C
    
    return mmd_est

def load_MSCI_data(dim, day_start, day_eval, day_end):
    data = {}
    for day in [2, 3, 4, 7]:
        data[day] = np.load(f"../data/full_cite_pcas_{dim}_day_{day}.npy")

    eval_data = data[day_eval]
    start_data = data[day_start]
    end_data = data[day_end]

    constant_scale = np.concatenate([start_data, end_data, eval_data]).std(axis=0).mean()

    eval_data_scaled = eval_data/constant_scale
    start_data_scaled = start_data/constant_scale
    end_data_scaled = end_data/constant_scale

    eval_data = torch.tensor(eval_data).float()
    start_data = torch.tensor(start_data_scaled).float()
    end_data = torch.tensor(end_data_scaled).float()

    X_sampler = TensorSampler(torch.tensor(start_data).float(), device="cpu")
    Y_sampler = TensorSampler(torch.tensor(end_data).float(), device="cpu")
    
    return X_sampler, Y_sampler, constant_scale, start_data, eval_data, end_data
