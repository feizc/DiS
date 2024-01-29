import torch 
import argparse
import os 
import torchvision

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler 
from torchvision import transforms
from glob import glob
from collections import OrderedDict
from copy import deepcopy
from PIL import Image
from tqdm import tqdm 
from torchvision.utils import save_image

from models_dis import DiS_models 
from diffusion import create_diffusion


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def main(args): 
    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True) 
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = DiS_models[args.model](
        img_size=args.image_size,
    ) 
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank]) 
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule 

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=False,
        transform=transform,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    update_ema(ema, model.module, decay=0) 
    model.train() 
    ema.eval()

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0

    for epoch in range(args.epochs): 
        sampler.set_epoch(epoch) 
        with tqdm(enumerate(loader), total=len(loader)) as tq:
            for data_iter_step, samples in tq: 
                x = samples[0].to(device) 
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device) 

                loss_dict = diffusion.training_losses(model, x, t)
                loss = loss_dict["loss"].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                update_ema(ema, model.module)

                running_loss += loss.item()
                log_steps += 1
                train_steps += 1

                tq.set_description('Epoch %i' % epoch) 
                tq.set_postfix(loss=running_loss / train_steps)
                # Save DiT checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path) 
                    dist.barrier()
                
                """
                # debug for ignored parameters 
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(name)
                """
                
        # eval 
        diffusion_eval = create_diffusion(str(250))
        n = 16
        z = torch.randn(n, 3, args.image_size, args.image_size, device=device)
        # Sample images:
        eval_samples = diffusion_eval.p_sample_loop(
            ema.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=None, progress=True, device=device
        )
        save_image(eval_samples, os.path.join(experiment_dir, "sample_" + str(epoch) + ".png"), nrow=4, normalize=True, value_range=(-1, 1))





if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiS_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 64, 32], default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0) 
    args = parser.parse_args()
    main(args)
