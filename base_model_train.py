#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   mlp_learning_an_image_pytorch.py
# @author Thomas Müller, NVIDIA
# @brief  Replicates the behavior of the CUDA mlp_learning_an_image.cu sample
#         using tiny-cuda-nn's PyTorch extension. Runs ~2x slower than native.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
import argparse
import datetime
import shutil
import commentjson as json
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import PIL
from custom_modules import MLP
from util import Image, SDFDataset, mean_relative_l2, mape_loss, save_mesh

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

# SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
# print(SCRIPTS_DIR)
# sys.path.insert(0, SCRIPTS_DIR)
from scripts.common import read_image, write_image, ROOT_DIR


# DATA_DIR = os.path.join(ROOT_DIR, "data")
# IMAGES_DIR = os.path.join(DATA_DIR, "images")


def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

    parser.add_argument("config", nargs="?", default="data/config_grid_mlp.json", help="JSON config for tiny-cuda-nn")
    parser.add_argument("--image", type=str, help="relative filepath to input image (for image fitting)")
    parser.add_argument("--mesh", type=str, help="relative filepath to input mesh OBJ (for SDF fitting)")
    parser.add_argument("--batch_sz", type=int, default=2**18, help="Number of samples per batch")
    parser.add_argument("--device", type=str, default="0", help="GPU index")
    parser.add_argument('--no-log-wandb', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")

    args = get_args()

    with open(args.config) as config_file:
        config = json.load(config_file)
    timestamp = time.time()
    timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
    device = torch.device(f"cuda:{args.device}")
    config["timestamp"] = timestamp
    assert args.image or args.mesh
    if args.image:
        config["training_image"] = args.image
    if args.mesh:
        config["training_mesh"] = args.mesh

    config["device"] = args.device
    config["batch_sz"] = args.batch_sz # do it here bc can't put math expressions in raw json (eg. 2**18)

    checkpoint_dir = f"checkpoints/{timestamp}"
    os.makedirs(checkpoint_dir)
    with open(f"{checkpoint_dir}/config.json", "w") as outfile:
        json.dump(config, outfile, indent=4)

    # save snapshot of current code just in case
    snapshot_dir = f"{checkpoint_dir}/code_snapshot"
    os.makedirs(snapshot_dir)
    shutil.copy("custom_modules.py", f"{snapshot_dir}/custom_modules.py")
    shutil.copy("util.py", f"{snapshot_dir}/util.py")
    shutil.copy("base_model_train.py", f"{snapshot_dir}/base_model_train.py")

    if args.image:
        image = Image(args.image, device)
        n_channels = image.data.shape[2]
        is_grayscale = (n_channels == 1)
        n_input_dims = 2
        # Variables for saving/displaying image results
        resolution = image.data.shape[0:2]
        img_shape = resolution + torch.Size([image.data.shape[2]])
        n_pixels = resolution[0] * resolution[1]
        half_dx =  0.5 / resolution[0] # half pixel size
        half_dy =  0.5 / resolution[1]
        xs = torch.linspace(half_dx, 1-half_dx, resolution[0], device=device)
        ys = torch.linspace(half_dy, 1-half_dy, resolution[1], device=device)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        path = f"{checkpoint_dir}/reference.png"
        print(f"Writing '{path}'... ", end="")
        if is_grayscale:
            PIL.Image.fromarray((255*image(xy, interpolate=False).reshape(img_shape).detach().cpu().numpy().squeeze()).astype(np.uint8)).save(path)
        else:
            write_image(path, image(xy).reshape(img_shape).detach().cpu().numpy())
        print("done.")

        # try:
        # 	batch = torch.rand([batch_size, 2], device=device, dtype=torch.float)
        # 	traced_image = torch.jit.trace(image, batch)
        # except:
            # If tracing causes an error, fall back to regular execution
        # print(f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular.")
        traced_image = image # for querying image 
        loss_fn = mean_relative_l2

    if args.mesh:
        print("SDF base model training")
        sdf_dataset = SDFDataset(args.mesh, device=device, clip_sdf=None, num_samples=config["batch_sz"], transformation_save_dir=checkpoint_dir)
        n_input_dims = 3
        n_channels = 1 # scalar SDF value
        shutil.copy(args.mesh, f"{checkpoint_dir}/reference.obj")

        loss_fn = mape_loss


    # encoding = tcnn.Encoding(n_input_dims=n_input_dims, encoding_config=config["encoding"], dtype=torch.float32)
    encoding = tcnn.Encoding(n_input_dims=n_input_dims, encoding_config=config["encoding"], dtype=torch.float32)
    encoding.to(device)
    # network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
    # construct MLP: R^input_dim --> R^output_dim
    network = MLP(input_dim=encoding.n_output_dims,
                output_dim=n_channels, 
                  hidden_dim=config["network"]["n_neurons"], 
                  num_hidden_layers=config["network"]["n_hidden_layers"],
                hidden_activation_fn=nn.ReLU(),
                output_activation_fn=None)
    network.to(device)

    model = torch.nn.Sequential(encoding, network)

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config["optimizer"]["learning_rate"],
                                 betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
                                 eps=config["optimizer"]["epsilon"])
    
    scheduler = None
    if config["optimizer"]["lr_scheduler_warmup_steps"] > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=config["optimizer"]["lr_scheduler_warmup_steps"])
    num_base_grid_params = sum(p.numel() for p in encoding.parameters())
    num_base_mlp_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    prev_time = time.perf_counter()
    interval = 10 # don't change
    print(f"Beginning optimization with {config["optimizer"]["n_steps"]} training steps.")
    ## training loop
    try:
        if not args.no_log_wandb:
            wandb.init(
                project="lora",
                config=config
            )

        print(f"Grid parameters: {num_base_grid_params}")
        print(f"MLP parameters: {num_base_mlp_params}")
        print(f"Total parameters: {trainable_params}")
        best_loss = float('inf')
        for i in range(config["optimizer"]["n_steps"]):
            if args.image:
                batch = torch.rand([args.batch_sz, 2], device=device, dtype=torch.float)
                targets = traced_image(batch)
            elif args.mesh:
                batch, targets = sdf_dataset.sample_batch()


            output = model(batch)
            loss = loss_fn(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            if i % 30 == 0:
                print(f"{i}: {loss.item()}")

            if not args.no_log_wandb and (i == 0 or i % int(interval//10) == 0):
                wandb.log({"step": i, "loss": loss.item()})

            if i % interval == 0: # reconstruct output
                loss_val = loss.item()
                torch.cuda.synchronize()
                elapsed_time = time.perf_counter() - prev_time
                print(f"Step#{i}: loss={loss_val} time={int(elapsed_time)}[s]")

                path = f"{checkpoint_dir}/{i}.png" if args.image else f"{checkpoint_dir}/{i}.obj"
                print(f"Writing '{path}'... ")
                with torch.no_grad():
                    if args.image:
                        if is_grayscale:
                            PIL.Image.fromarray((255*model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy().squeeze()).astype(np.uint8)).save(path)
                        else:
                            write_image(path, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
                    elif args.mesh and i > 1000:
                        save_mesh(path, model, device, resolution=256)
                if loss_val < best_loss:
                    print(f"decreased {best_loss}-->{loss_val}")
                    best_loss = loss_val
                    print(f"saving model at {checkpoint_dir}")
                    torch.save({
                        'step': i,
                        'model_state_dict': model.state_dict(),
                        'loss': loss_val,
                    }, f"{checkpoint_dir}/base_model_weights.pt")
                print('done')
                # Ignore the time spent saving the image
                prev_time = time.perf_counter()

                if i > 0 and interval < 1000:
                    interval *= 10

        result_filename = f"{checkpoint_dir}/trained_model_output.png" if args.image else f"{checkpoint_dir}/trained_model_output.obj"
        print(f"Writing '{result_filename}'... ", end="")
        with torch.no_grad():
            if args.image:
                if is_grayscale:
                    PIL.Image.fromarray((255*model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy().squeeze()).astype(np.uint8)).save(path)
                else:
                    write_image(result_filename, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
            elif args.mesh:
                save_mesh(result_filename, model, device, resolution=512)
        print("done.")
    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Exiting...")
        tcnn.free_temporary_memory()
    tcnn.free_temporary_memory()
