
import argparse
import datetime
import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from custom_modules import MLP
from util import SDFDataset, extract_fields, listdir_by_time, mape_loss, mean_relative_l2, mean_squared_error, save_mesh, sort_files_with_number_at_end, Image, peak_signal_noise_ratio
import wandb
from scripts.common import write_image

try:
    import tinycudann as tcnn
except ImportError:
    print("This sample requires the tiny-cuda-nn extension for PyTorch.")
    print("You can install it by running:")
    print("============================================================")
    print("tiny-cuda-nn$ cd bindings/torch")
    print("tiny-cuda-nn/bindings/torch$ python setup.py install")
    print("============================================================")
    sys.exit()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    parser = argparse.ArgumentParser(description="Post-process low-rank reconstruction using SVD")
    parser.add_argument("finetuned_checkpoint_dir", type=str, help="relative checkpoint directory of the trained lora or fully finetuned model (output of train_lora.py with full FT)")
    parser.add_argument("rank", type=int, help="(max) rank of rank-constrained weight difference")
    # parser.add_argument("--finetuned", action='store_true', help="set flag if the checkpoint was for a fully finetuned model (no LoRA)") # TODO infer from saved ckpt name
    # parser.add_argument("--image", action='store_true', help="set if input type is a single image")
    # parser.add_argument("--mesh", action='store_true', help="set if input type is a mesh")
    parser.add_argument("--device", type=str, default="0", help="GPU index")

    args = parser.parse_args()
    return args

def list_files_with_extension(directory, extension):
    """Lists all files with a given extension in a directory and its subdirectories."""
    return glob.glob(f"{directory}/**/*.{extension}", recursive=True)

if __name__ == "__main__":
    print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")
    args = get_args()
    torch.set_printoptions(precision=7)

    lora_config = json.load(open(f"{args.finetuned_checkpoint_dir}/lora_config.json"))
    device = torch.device(f"cuda:{args.device}")
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    eval_dir = f"eval/{args.finetuned_checkpoint_dir}/svd_postprocess_rank{args.rank}_{timestamp}/"
    os.makedirs(eval_dir)

    # determine whether to use lora or fully finetuned params
    checkpoint_files = list_files_with_extension(args.finetuned_checkpoint_dir, ".pt")
    use_lora_weights = True # don't touch
    for ckpt in checkpoint_files:
        print(ckpt)
        assert (ckpt == "lora_weights.pt") ^ (ckpt == "finetuned_weights.pt")
        if ckpt == "finetuned_weights.pt": 
            use_lora_weights = False
            break
    # assert (args.image) ^ (args.mesh)

    # if args.image:
    reference_path = f"{args.finetuned_checkpoint_dir}/reference.png"
    shutil.copy(reference_path, f"{eval_dir}/reference.png")

    # else:
        # print("SDF input")
        # train_video_lora = False
        # reference_paths = [f"{args.finetuned_checkpoint_dir}/reference.obj"]


            
    #  base model dir is one dir up from lora ckpt 
    print(args.finetuned_checkpoint_dir)
    base_model_checkpoint_dir = Path(args.finetuned_checkpoint_dir).parent
    # base_model_checkpoint = torch.load(f"{base_model_checkpoint_dir}/base_model_weights.pt", weights_only=True)
    print("reference image: ", reference_path)

    psnr_per_frame = []
    # for frame_idx, reference_path in enumerate(reference_paths):
    save_dirname = "" 

    # data prep
    reference_image = Image(reference_path, device)
    n_channels = 2
    output_extension = "png" # "jpg"
    n_channels = reference_image.data.shape[2]
    
    # Variables for saving/displaying image results
    resolution = reference_image.data.shape[0:2]
    img_shape = resolution + torch.Size([reference_image.data.shape[2]])
    n_pixels = resolution[0] * resolution[1]
    half_dx =  0.5 / resolution[0] # half pixel size
    half_dy =  0.5 / resolution[1]
    xs = torch.linspace(half_dx, 1-half_dx, resolution[0], device=device)
    ys = torch.linspace(half_dy, 1-half_dy, resolution[1], device=device)
    xv, yv = torch.meshgrid([xs, ys], indexing="ij")
    xy = torch.stack((yv.flatten(), xv.flatten())).t()
    # path = f"{lora_checkpoint_dir}/{save_dirname}/reference.{output_extension}"
    # print(f"Writing '{path}'... ", end="")
    # write_image(path, image(xy).reshape(img_shape).detach().cpu().numpy())
    # print("done.")
    spatial_input_dim = 2
    

        

    # load pre-trained base model
    base_model_config = json.load(open(f"{base_model_checkpoint_dir}/config.json"))
    encoding = tcnn.Encoding(n_input_dims=spatial_input_dim, encoding_config=base_model_config["encoding"], dtype=torch.float32)
    # construct MLP: R^input_dim --> R^output_dim
    mlp = MLP(input_dim=encoding.n_output_dims,
                output_dim=n_channels, 
                hidden_dim=base_model_config["network"]["n_neurons"], 
                num_hidden_layers=base_model_config["network"]["n_hidden_layers"],
                hidden_activation_fn=nn.ReLU(),
                output_activation_fn=None)

    model = torch.nn.Sequential(encoding, mlp)
    base_model_checkpoint = torch.load(f"{base_model_checkpoint_dir}/base_model_weights.pt", weights_only=True)
    model.load_state_dict(base_model_checkpoint['model_state_dict'])
    model.to(device)
    # cache base weights
    base_weights = [] # list of linear layer weights
    for layer in model[1].layers:
        if isinstance(layer, nn.Linear):
            base_weights.append(torch.clone(layer.weight))

    # separate grid encoding and mlp
    encoding = model[0]
    for params in encoding.parameters():
        params.requires_grad = False
    encoding.to(device)
    mlp = model[1]
    mlp.to(device)


    # load & cache fully finetuned weights
    mlp.load_state_dict(torch.load(f"{args.finetuned_checkpoint_dir}/{save_dirname}/finetuned_weights.pt", weights_only=True)['model_state_dict'])
    finetuned_weights = [] # list of linear layer weights
    for layer in mlp.layers:
        if isinstance(layer, nn.Linear):
            finetuned_weights.append(layer.weight)
    finetuned_reconstruction = mlp(encoding(xy)).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
    write_image(f"{eval_dir}/full_finetuning_reconstruction.png", finetuned_reconstruction)

    # look at SVD singular vals of difference
    rank_constrained_delta_Ws = [] # one matrix per layer
    for i, (base_weight, finetuned_weight) in enumerate(zip(base_weights, finetuned_weights)):
        delta_W = finetuned_weight - base_weight
        # U, s, V = np.linalg.svd(delta_W.detach().numpy())
        # print("layer ", i)
        # # print(delta_W)
        # # print(s)
        # # continue

        # # Create bar plot of delta W singular values
        # plt.figure(figsize=(10, 6))
        # plt.bar(range(len(s)), s)
        # plt.xlabel("Singular Value Index")
        # plt.ylabel("Singular Value")
        # plt.title(f"Layer {i}: Singular Values of delta W := finetuned_weight - base_weight")
        # plt.xticks(np.arange(0, len(s), 20))
        # plt.grid(axis='y')
        # # plt.show()
        # # save plot of SVs
        # plt.savefig(f"{args.finetuned_checkpoint_dir}/layer{i}_delta_W_singular_vals.png", dpi=200)
        # print("rank: ", s[s>1e-3].shape[0])


        # # Create bar plot of base singular values
        # U, s, V = np.linalg.svd(base_weight.detach().numpy())
        # plt.figure(figsize=(10, 6))
        # plt.bar(range(len(s)), s)
        # plt.xlabel("Singular Value Index")
        # plt.ylabel("Singular Value")
        # plt.title(f"Layer {i}: Singular Values of base weight")
        # plt.xticks(np.arange(0, len(s), 20))
        # plt.grid(axis='y')
        # # plt.show()
        # # save plot of SVs
        # plt.savefig(f"{args.finetuned_checkpoint_dir}/layer{i}_base_W_singular_vals.png", dpi=200)


        # line plot of cumulative percent variance explained
        # U, s, V = np.linalg.svd(delta_W.detach().numpy())
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)

        # truncate to rank-k
        k = args.rank
        U_k = U[:, :k]     # (d_out x k)
        S_k = S[:k]        # (k,)
        Vh_k = Vh[:k, :]   # (k x d_in)

        # Reconstruct the rank-k approximation
        delta_W_rank_k = U_k @ torch.diag(S_k) @ Vh_k
        rank_constrained_delta_Ws.append(delta_W_rank_k)
        
        s = S.detach().cpu().numpy()
        explained_variance_ratio = (s*s) / np.sum(s**2) # normalized, [num_singular_vals,]
        cumulative_EVR = np.cumsum(explained_variance_ratio)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(s)), cumulative_EVR)
        plt.ylim(0, 1)
        plt.xlabel(f"Singular Value Index ({len(s)} total)")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title(f"Layer {i}: Cumulative Explained Variance Ratio")
        plt.xticks(np.arange(0, len(s), 20))
        plt.yticks(np.arange(0, 1 + 0.1, 0.1))
        plt.grid(axis='y')
        # plt.show()
        # save plot of SVs
        plt.savefig(f"{eval_dir}/layer{i}_delta_W_cumulative_EVR.png", dpi=200)
        plt.close()

        # save rank-constrained weight difference
        torch.save(delta_W_rank_k, f'{eval_dir}/layer{i}_delta_W_truncated_rank{k}.pt')


        # TEMP (full rank)
        if not torch.allclose( U@torch.diag(S)@Vh, delta_W):
            print("max entrywise difference (true delta_W vs SVD reconstruction)")
            print(torch.abs(U@torch.diag(S)@Vh - delta_W).max().item())

    # add weight difference to base weights
    model.load_state_dict(base_model_checkpoint['model_state_dict'])
    idx = 0
    with torch.no_grad():
        for layer in mlp.layers:
            if isinstance(layer, nn.Linear):
                layer.weight.copy_(layer.weight + rank_constrained_delta_Ws[idx])
                idx += 1

        # reconstruct output image
        reconstruction = model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
        save_path = f"{eval_dir}/post_process_rank{k}_reconstruction.png"
        write_image(save_path, reconstruction)
        print("saved at: ", save_path)

        # compute metrics
        results = {}
        reconstructed_image = Image(save_path, device)  # load from disk
        psnr = peak_signal_noise_ratio(reference_image.orig_data_npy, reconstructed_image.orig_data_npy)
        mse = mean_squared_error(reference_image.orig_data_npy, reconstructed_image.orig_data_npy)

        results['psnr'] = psnr
        results['mse'] = mse
        print(f"PSNR: {psnr}")
        print(f"MSE: {mse}")
        with open(f"{eval_dir}/results.json", "w") as outfile:
            json.dump(results, outfile, indent=4)