import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

import argparse
import datetime
import json
import random
import shutil
import sys
from matplotlib import pyplot as plt
import numpy as np
import time
import torch
import copy
import torch.nn as nn
from custom_modules import MLP, LoRA_MLP, extract_linear_layers
from util import DataSampler, SDFDataset, listdir_by_time, edit_region_percentage_to_normalized_bounds, mape_loss, mean_relative_l2, save_mesh, sort_files_with_number_at_end, Image
# import wandb
from scripts.common import read_image, write_image
# try:
#     import tinycudann as tcnn
# except ImportError:
#     print("This sample requires the tiny-cuda-nn extension for PyTorch.")
#     print("You can install it by running:")
#     print("============================================================")
#     print("tiny-cuda-nn$ cd bindings/torch")
#     print("tiny-cuda-nn/bindings/torch$ python setup.py install")
#     print("============================================================")
#     sys.exit()
from typing import Callable

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

    # parser.add_argument("base_model_checkpoint_dir", type=str, help="relative checkpoint directory of pretrained/base model")
    # parser.add_argument("--image", type=str, help="relative filepath to perturbed input image (for image fitting)")
    # parser.add_argument("--mesh", type=str, help="relative filepath to perturbed mesh (for SDF fitting)")
    # parser.add_argument("--frames", type=str, help="directory containing video frame images: frame_xxxxx.png")
    # parser.add_argument("--device", type=str, default="0", help="GPU index")
    # parser.add_argument('--no-log-wandb', action='store_true')

    args = parser.parse_args()
    return args

def get_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return next(module.buffers()).device

def train_lora_regression(
        base_nf: nn.Sequential, 
        target_sampler: DataSampler, 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        lora_rank: int, 
        learning_rate=5e-3, 
        batch_size=2**18, 
        max_n_steps=30000, 
        lr_scheduler_warmup_steps=7000, 
        log_interval=100,
        convergence_patience=15,
        save_dir="",
    ):
    """
    Trains LoRAs for every linear layer of the given base neural field. Returns a new neural field with a LoRA applied and (optionally) the weights of each LoRA
    
    Args:
        - base_nf: MLP implemented as a nn.Sequential (containing nn.Linear layers, activation functions, optional input positional encoding)
        - target_sampler: a DataSampler that must be compatible with base_nf (i.e. input/target dimensions must match)
        - loss_fn: callable function (output,target) |-> loss scalar
        - lora_rank: Desired maximum rank of each LoRA
        - learning_rate: for ADAM optimizer
        - batch_size: num samples per step
        - max_n_steps: max number of training steps
        - lr_scheduler_warmup_steps: number of warmup steps for the learning rate scheduler. No warmup if 0.
        - convergence_patience: number of log_intervals of no improvement after which training is terminated early
        - save_dir: if provided, then weights and output reconstructions will be saved there
    
    Returns:
        - lora_weights: LoRA weight tensors, one per linear layer of the base model. Returns the iterate with the lowest loss.
        - lora_nf: nn.Sequential neural field with LoRA applied to every linear layer. Returns the iterate with the lowest loss.
    """
    # make sure that base_nf and target_sampler are compatible
    linear_layers = extract_linear_layers(base_nf)
    base_output_dim = linear_layers[-1].out_features
    _, dummy_targets = target_sampler.sample_batch(batch_size)
    assert dummy_targets.shape[-1] == base_output_dim, "base_nf and target_sampler are incompatible: output size mismatch"
    assert isinstance(base_nf, nn.Sequential)

    lora_nf = LoRA_MLP(base_nf, lora_rank)
    device = get_device(base_nf)
    print(f"training LoRA on {device}")
    trainable_params = sum(p.numel() for p in lora_nf.parameters() if p.requires_grad)
    print(f"# trainable LoRA parameters: {trainable_params}")

    optimizer = torch.optim.Adam(lora_nf.parameters(), lr=learning_rate) # only fine-tune mlp
    # use learning rate scheduler if specified
    scheduler = None        
    if lr_scheduler_warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=lr_scheduler_warmup_steps)
    best_loss = float('inf')
    best_lora_nf = copy.deepcopy(lora_nf)
    prev_time = time.perf_counter()
    periods_no_improve = 0 # for early stopping

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(max_n_steps):
        batch_inputs, batch_targets = target_sampler.sample_batch(batch_size) # [B,in], [B,out]
        output = lora_nf(batch_inputs.to(device))
        loss = loss_fn(output, batch_targets.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        # if i % 30 == 0: # TEMP
        #     print(f"{i}: {loss.item()}")

        if i % log_interval == 0: # reconstruct output
            curr_loss = loss.item()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={curr_loss:.7f} time={int(elapsed_time)}[s]")

            if curr_loss < best_loss:
                print(f"decreased {best_loss:.6f}-->{curr_loss:.6f}")
                best_loss = curr_loss
                periods_no_improve = 0 # reset
                best_lora_nf = copy.deepcopy(lora_nf)
                # weights_filename = "lora_weights.pt"
                # print(f"saving model at {checkpoint_dir}/{save_dirname}")
                # torch.save({ # TEMP
                #     'step': i,
                #     'model_state_dict': mlp.lora_mlps.state_dict(),
                #     'loss': curr_loss,
                # }, f"{checkpoint_dir}/{save_dirname}/{weights_filename}")
            else:
                # early stopping if no improvement for several epochs
                periods_no_improve += 1
                if periods_no_improve >= convergence_patience:
                    print(f"Early stopping at step {i} with loss {curr_loss}")
                    break
            prev_time = time.perf_counter()
            if i > 0 and log_interval < 1000: # TEMP
                log_interval *= 10
    if save_dir:
        target_sampler.save_model_output(best_lora_nf, save_path=f"{save_dir}/lora_nf_best")
    
    return best_lora_nf.get_lora_weights(), best_lora_nf.as_sequential()

def train_base_model(
        base_nf: nn.Sequential, 
        data_sampler: DataSampler, 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        learning_rate=5e-3, 
        batch_size=2**18, 
        max_n_steps=30000, 
        lr_scheduler_warmup_steps=7000, 
        log_interval=100,
        convergence_patience=15,
        save_dir="",
    ):
    """
    Trains the given base neural field to regress samples from data_sampler.

    Returns a copy of base_nf at the best training iteration
    """
    device = get_device(base_nf)
    trainable_params = sum(p.numel() for p in base_nf.parameters() if p.requires_grad)
    print(f"training base model on {device}")
    print(f"# trainable base model parameters: {trainable_params}")
    
    optimizer = torch.optim.Adam(base_nf.parameters(), lr=learning_rate) # only fine-tune mlp
    # use learning rate scheduler if specified
    scheduler = None        
    if lr_scheduler_warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=lr_scheduler_warmup_steps)
    best_loss = float('inf')
    best_model = copy.deepcopy(base_nf)
    prev_time = time.perf_counter()
    periods_no_improve = 0 # for early stopping

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(max_n_steps):
        batch_inputs, batch_targets = data_sampler.sample_batch(batch_size) # [B,in], [B,out]
        output = base_nf(batch_inputs.to(device))
        loss = loss_fn(output, batch_targets.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        # if i % 30 == 0: # TEMP
        #     print(f"{i}: {loss.item()}")

        if i % log_interval == 0: # reconstruct output
            curr_loss = loss.item()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={curr_loss:.7f} time={int(elapsed_time)}[s]")

            if curr_loss < best_loss:
                print(f"decreased {best_loss:.6f}-->{curr_loss:.6f}")
                best_loss = curr_loss
                periods_no_improve = 0 # reset
                best_model = copy.deepcopy(base_nf)
                # weights_filename = "lora_weights.pt"
                # print(f"saving model at {checkpoint_dir}/{save_dirname}")
                # torch.save({ # TEMP
                #     'step': i,
                #     'model_state_dict': mlp.lora_mlps.state_dict(),
                #     'loss': curr_loss,
                # }, f"{checkpoint_dir}/{save_dirname}/{weights_filename}")
            else:
                # early stopping if no improvement for several epochs
                periods_no_improve += 1
                if periods_no_improve >= convergence_patience:
                    print(f"Early stopping at step {i} with loss {curr_loss}")
                    break
            prev_time = time.perf_counter()
            if i > 0 and log_interval < 1000: # TEMP
                log_interval *= 10
    if save_dir:
        data_sampler.save_model_output(best_model, save_path=f"{save_dir}/base_nf")
    
    return best_model

def image_demo():
    save_dir = "checkpoints/minimal"
    base_image_path= "data/images/table/table_before.png"
    target_image_path= "data/images/table/table_after.png"
    device = "cpu"

    # set up base model
    base_nf = nn.Sequential(
        nn.Linear(2, 10), # xy input TODO pos enc
        # nn.ReLU(),        
        nn.Linear(10, 10), 
        nn.ReLU(),        
        nn.Linear(10, 3), # RGB output 
    )
    base_image_sampler = Image(base_image_path, device)
    loss_fn = mean_relative_l2 
    base_nf = train_base_model(base_nf, base_image_sampler, loss_fn, save_dir=save_dir, max_n_steps=100)
    
    #########################
    # train lora
    target_image_sampler = Image(target_image_path, device)
    lora_rank = 3

    breakpoint()
    lora_weights, lora_nf = train_lora_regression(base_nf, target_image_sampler, loss_fn, lora_rank, save_dir=save_dir, max_n_steps=100) # TEMP save dir
    breakpoint()

# ### LORA EDITING ###
lora_config = { # shared across different frames
    "learning_rate": 5e-3, #1e-4, ### DEFAULTS: 1e-4 for finetuning, 5e-3 for LoRA 
    "rank": 64,
    "batch_size": 2**18,
    "n_steps": 30000,
    "lr_scheduler_warmup_steps": 7000, 
    "convergence_patience": 15, # num of epochs of no loss improvement before declaring convergence
    "full_finetuning": False,
    # "num_active_layers": "all",  # counting from the last (output) layer backward. e.g. if n, then the last n layers (incl output) are active
} # TODO: make config json
# ##################

if __name__ == "__main__":
    print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")
    args = get_args()
    torch.set_printoptions(precision=7)
    try: 
        image_demo()
    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Exiting...")











    # exit()

    # # get frames
    # if args.frames:
    #     print("video input")
    #     training_input_paths = listdir_by_time(args.frames, endswith=".png")
    #     # order by frame number
    #     training_input_paths = sort_files_with_number_at_end(training_input_paths)
    #     train_video_lora = True
    # elif args.image:
    #     print("single image input")
    #     train_video_lora = False
    #     training_input_paths = [args.image]
    # else:
    #     assert args.mesh
    #     print("SDF input")
    #     train_video_lora = False
    #     training_input_paths = [args.mesh]


    # # save lora config as json
    # timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    # lora_checkpoint_dir = f"{args.base_model_checkpoint_dir}/lora_{timestamp}/"
    # lora_config["timestamp"] = timestamp
    # lora_config["device"] = args.device
    # lora_config["base_model_checkpoint_dir"] = args.base_model_checkpoint_dir
    # assert (args.image is not None) \
    #         ^ (args.mesh is not None) \
    #         ^ (args.frames is not None)

    # if args.image:
    #     lora_config["training_image"] = args.image
    # if args.frames:
    #     lora_config["training_frames"] = args.frames
    # if args.mesh:
    #     lora_config["training_mesh"] = args.mesh
    
    # os.makedirs(lora_checkpoint_dir)
    # with open(f"{lora_checkpoint_dir}/lora_config.json", "w") as outfile:
    #     json.dump(lora_config, outfile, indent=4)
    # with open(f"{lora_checkpoint_dir}/command.txt", "w") as file:
    #     file.write('python '+ ' '.join(sys.argv))

    # # save snapshot of current code just in case
    # snapshot_dir = f"{lora_checkpoint_dir}/code_snapshot"
    # os.makedirs(snapshot_dir)
    # shutil.copy("train_lora.py", f"{snapshot_dir}/train_lora.py")
    # shutil.copy("base_model_train.py", f"{snapshot_dir}/train.py")
    # shutil.copy("custom_modules.py", f"{snapshot_dir}/modules.py")
    # shutil.copy("util.py", f"{snapshot_dir}/util.py")
    # device = torch.device(f"cuda:{args.device}")

    # try:
    #     if not args.no_log_wandb:
    #         wandb.init(
    #             project="lora",
    #             config=lora_config
    #         )
            
    #     prev_frame_checkpoint_paths =[]
    #     base_model_checkpoint = torch.load(f"{args.base_model_checkpoint_dir}/base_model_weights.pt", weights_only=True)
    #     start_epoch = base_model_checkpoint["step"]
    #     print("ordered inputs: ", training_input_paths)
    #     logging_step_offset = base_model_checkpoint["step"]
    #     for frame_idx, input_path in enumerate(training_input_paths):
    #         if train_video_lora:
    #             # TEMP: start with frame 1 instead of frame 0 (which was used to train base model)
    #             if frame_idx == 0: 
    #                 continue
    #             else:
    #                 print(20*"=")
    #                 print(f"\t FRAME {frame_idx}")
    #                 save_dirname = f"lora_frame{frame_idx}" # make subdirectory for every frame
    #                 os.makedirs(f"{lora_checkpoint_dir}/{save_dirname}")
    #         else:
    #             save_dirname = "" 


    #         # data prep
    #         if args.image or args.frames:
    #             # image data 
    #             print("path: ", input_path)
    #             image = Image(input_path, device)
    #             n_channels = 2
    #             output_extension = "png" # "jpg"
    #             n_channels = image.data.shape[2]
                
    #             # Variables for saving/displaying image results
    #             resolution = image.data.shape[0:2]
    #             img_shape = resolution + torch.Size([image.data.shape[2]])
    #             n_pixels = resolution[0] * resolution[1]
    #             half_dx =  0.5 / resolution[0] # half pixel size
    #             half_dy =  0.5 / resolution[1]
    #             xs = torch.linspace(half_dx, 1-half_dx, resolution[0], device=device)
    #             ys = torch.linspace(half_dy, 1-half_dy, resolution[1], device=device)
    #             xv, yv = torch.meshgrid([xs, ys], indexing="ij")
    #             xy = torch.stack((yv.flatten(), xv.flatten())).t()
    #             path = f"{lora_checkpoint_dir}/{save_dirname}/reference.{output_extension}"
    #             print(f"Writing '{path}'... ", end="")
    #             write_image(path, image(xy).reshape(img_shape).detach().cpu().numpy())
    #             print("done.")
    #             spatial_input_dim = 2
    #             loss_fn = mean_relative_l2
    #         else:
    #             # SDF data
    #             assert args.mesh
    #             sdf_dataset = SDFDataset(args.mesh, device=device, clip_sdf=None, num_samples=lora_config["batch_size"], transformation_save_dir=args.base_model_checkpoint_dir)
    #             shutil.copy(args.mesh, f"{lora_checkpoint_dir}/{save_dirname}/reference.obj")
    #             n_channels = 1
    #             spatial_input_dim = 3
    #             output_extension = "obj"
    #             loss_fn = mape_loss

                

    #         # load pre-trained base model
    #         base_model_config = json.load(open(f"{args.base_model_checkpoint_dir}/config.json"))
    #         encoding = tcnn.Encoding(n_input_dims=spatial_input_dim, encoding_config=base_model_config["encoding"], dtype=torch.float32)
    #         # construct MLP: R^input_dim --> R^output_dim
    #         mlp = MLP(input_dim=encoding.n_output_dims,
    #                     output_dim=n_channels, 
    #                     hidden_dim=base_model_config["network"]["n_neurons"], 
    #                     num_hidden_layers=base_model_config["network"]["n_hidden_layers"],
    #                     hidden_activation_fn=nn.ReLU(),
    #                     output_activation_fn=None)

    #         model = torch.nn.Sequential(encoding, mlp)
    #         base_model_checkpoint = torch.load(f"{args.base_model_checkpoint_dir}/base_model_weights.pt", weights_only=True)
    #         model.load_state_dict(base_model_checkpoint['model_state_dict'])
    #         num_base_grid_params = sum(p.numel() for p in encoding.parameters())
    #         num_base_mlp_params = sum(p.numel() for p in mlp.parameters())

    #         # separate grid encoding and mlp
    #         encoding = model[0]
    #         for params in encoding.parameters():
    #             params.requires_grad = False
    #         encoding.to(device)

    #         mlp = model[1]

    #         if not lora_config["full_finetuning"]:
    #             # use lora
    #             mlp.initialize_lora(rank=lora_config["rank"], prev_frame_lora_paths=prev_frame_checkpoint_paths) # also freezes trained base model's weights
    #             forward_fn = lambda input_pos : mlp.forward_with_lora(input_pos, encoding(input_pos))
    #             if train_video_lora:
    #                 # store path to current weights (to be trained) for next frame
    #                 prev_frame_checkpoint_paths.append(f"{lora_checkpoint_dir}/{save_dirname}/lora_weights.pt")
    #         else:
    #             # load prev frame's mlp weights
    #             forward_fn = lambda input_pos : mlp(encoding(input_pos))
    #             if prev_frame_checkpoint_paths:
    #                 mlp.load_state_dict(torch.load(prev_frame_checkpoint_paths[-1], weights_only=True)['model_state_dict'])
    #             prev_frame_checkpoint_paths.append(f"{lora_checkpoint_dir}/{save_dirname}/finetuned_weights.pt")
    #         mlp.to(device)


    #         # train lora mlp for this frame
    #         print(f"Number of base model parameters (grid): {num_base_grid_params}")
    #         print(f"Number of base model parameters (mlp): {num_base_mlp_params}")
    #         trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    #         print(f"Trainable parameters: {trainable_params}")

    #         path = f"{lora_checkpoint_dir}/{save_dirname}/before_training.{output_extension}"
    #         print(f"Writing '{path}'... ")
    #         with torch.no_grad():
    #             if args.mesh:
    #                 save_mesh(path, forward_fn, device, resolution=512)
    #             else:
    #                 write_image(path, forward_fn(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
            
    #         optimizer = torch.optim.Adam(mlp.parameters(), lr=lora_config["learning_rate"]) # only fine-tune mlp
    #         # use learning rate scheduler if specified
    #         scheduler = None        
    #         if lora_config["lr_scheduler_warmup_steps"] > 0:
    #             scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=lora_config["lr_scheduler_warmup_steps"])
    #         best_loss = float('inf')
    #         interval = 10
    #         prev_time = time.perf_counter()
    #         periods_no_improve = 0

    #         for i in range(lora_config["n_steps"]):
    #             if args.mesh:
    #                 batch_inputs, targets = sdf_dataset.sample_batch() 
    #             else:
    #                 batch_inputs = torch.rand([lora_config["batch_size"], 2], device=device, dtype=torch.float)
    #                 targets = image(batch_inputs) # query image
                
    #             output = forward_fn(batch_inputs)
    #             loss = loss_fn(output, targets)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             if scheduler:
    #                 scheduler.step()

    #             if i % 30 == 0:
    #                 print(f"{i}: {loss.item()}")

    #             if (i % int(interval//10) == 0) and not args.no_log_wandb:
    #                 wandb.log({"step": i + logging_step_offset, "loss": loss.item()})

    #             if i % interval == 0: # reconstruct output
    #                 loss_val = loss.item()
    #                 torch.cuda.synchronize()
    #                 elapsed_time = time.perf_counter() - prev_time
    #                 print("-"*30)
    #                 print(f"Step#{i}: loss={loss_val:.7f} time={int(elapsed_time)}[s]")


    #                 path = f"{lora_checkpoint_dir}/{save_dirname}/{i}.{output_extension}"
    #                 print(f"==> Writing '{path}'... ")
    #                 with torch.no_grad():
    #                     if args.mesh:
    #                         save_mesh(path, forward_fn, device, resolution=256)
    #                     else:
    #                         write_image(path, forward_fn(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
    #                 print("<== done writing.")
    #                 if loss_val < best_loss:
    #                     print(f"decreased {best_loss:.6f}-->{loss_val:.6f}")
    #                     best_loss = loss_val
    #                     periods_no_improve = 0 # reset
    #                     weights_filename = "finetuned_weights.pt" if lora_config["full_finetuning"] else "lora_weights.pt"
    #                     print(f"saving model at {lora_checkpoint_dir}/{save_dirname}")
    #                     torch.save({
    #                         'step': i,
    #                         'model_state_dict': mlp.state_dict() if lora_config["full_finetuning"] else mlp.lora_mlps.state_dict(),
    #                         'loss': loss_val,
    #                     }, f"{lora_checkpoint_dir}/{save_dirname}/{weights_filename}")
    #                 else:
    #                     # early stopping if no improvement for several epochs
    #                     periods_no_improve += 1
    #                     if periods_no_improve >= lora_config["convergence_patience"]:
    #                         print(f"Early stopping at step {i} with loss {loss_val}")
    #                         break
    #                 print("done.")

    #                 # Ignore the time spent saving the output
    #                 prev_time = time.perf_counter()

    #                 if i > 0 and interval < 1000:
    #                     interval *= 10

    #         logging_step_offset += i
    #         result_path = f"{lora_checkpoint_dir}/{save_dirname}/trained_model_output.{output_extension}"
    #         print(f"==> Writing '{result_path}'... ")
    #         with torch.no_grad():
    #             if args.mesh:
    #                 save_mesh(result_path, forward_fn, device, resolution=512)
    #             else:
    #                 write_image(result_path, forward_fn(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
    #         print("<== done writing.")

    # except KeyboardInterrupt:
    #     print("\nCtrl+C pressed. Exiting...")
    #     tcnn.free_temporary_memory()
    # tcnn.free_temporary_memory()

    # exit()