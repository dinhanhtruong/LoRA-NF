## this script is for reconstructing an image or SDF from the best 
## checkpoint (LoRA or full finetuning) and computing metrics against the target image


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
from custom_modules import MLP
from util import SDFDataset, extract_fields, listdir_by_time, mape_loss, mean_relative_l2, mean_squared_error, save_mesh, sort_files_with_number_at_end, Image, peak_signal_noise_ratio
import wandb
from scripts.common import write_image, write_image_imageio
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
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")
    parser.add_argument("lora_or_finetuned_checkpoint_dir", type=str, help="relative checkpoint directory of the trained lora or fully finetuned model (output of train_lora.py)")
    # parser.add_argument("--finetuned", action='store_true', help="set flag if the checkpoint was for a fully finetuned model (no LoRA)") # TODO infer from saved ckpt name
    parser.add_argument("--image", action='store_true', help="set if input type is a single image")
    parser.add_argument("--mesh", action='store_true', help="set if input type is a mesh")
    parser.add_argument("--video", action='store_true', help="set if input type is a video (sequence of frames)")
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

    lora_config = json.load(open(f"{args.lora_or_finetuned_checkpoint_dir}/lora_config.json"))
    device = torch.device(f"cuda:{args.device}")
    print(device)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    eval_dir = f"eval/{args.lora_or_finetuned_checkpoint_dir}/eval_{timestamp}/"
    os.makedirs(eval_dir)

    # determine whether to use lora or fully finetuned params
    checkpoint_files = list_files_with_extension(args.lora_or_finetuned_checkpoint_dir, ".pt")
    use_lora_weights = True
    for ckpt in checkpoint_files:
        print(ckpt)
        assert (ckpt == "lora_weights.pt") ^ (ckpt == "finetuned_weights.pt")
        if ckpt == "finetuned_weights.pt": 
            use_lora_weights = False
            break
    
    assert (args.image) ^ (args.mesh)  ^ (args.video)

    # get frames
    if args.video:
        print("video input")
        # training_input_paths = listdir_by_time(args.frames, endswith=".png")
        # order by frame number
        reference_paths = sort_files_with_number_at_end(os.listdir(args.lora_or_finetuned_checkpoint_dir))
        # frame dirs start with "lora_frame"
        reference_paths = [f"{args.lora_or_finetuned_checkpoint_dir}/{dir}" for dir in reference_paths if dir.startswith("lora_frame")]
        reference_paths = [f"{path}/reference.png" for path in reference_paths] # jpg for highway video
        train_video_lora = True
    elif args.image:
        print("single image input")
        train_video_lora = False
        reference_paths = [f"{args.lora_or_finetuned_checkpoint_dir}/reference.png"]
    else:
        assert args.mesh
        print("SDF input")
        train_video_lora = False
        reference_paths = [f"{args.lora_or_finetuned_checkpoint_dir}/reference.obj"]
        # NOTE: forgot to copy the reference mesh in base_model_train.py so will need to manually copy it to the trained LoRA/finetuned dir


    
            
    prev_frame_checkpoint_paths =[]
    #  base model dir is one dir up from lora ckpt 
    print(args.lora_or_finetuned_checkpoint_dir)
    base_model_checkpoint_dir = Path(args.lora_or_finetuned_checkpoint_dir).parent
    # base_model_checkpoint = torch.load(f"{base_model_checkpoint_dir}/base_model_weights.pt", weights_only=True)
    print("ordered inputs: ", reference_paths)

    psnr_per_frame = []
    for frame_idx, reference_path in enumerate(reference_paths):
        if train_video_lora:
            # TEMP: start with frame 1 instead of frame 0 (which was used to train base model)
            print(20*"=")
            print(f"\t FRAME {frame_idx+1}")
            save_dirname = f"lora_frame{frame_idx+1}" # subdirectory for every frame
        else:
            save_dirname = "" 


        # data prep
        if args.image or args.video:
            # image data 
            print("path: ", reference_path)
            reference_image = Image(reference_path, device)
            n_channels = 2
            output_extension = "png" # "jpg"
            n_channels = reference_image.data.shape[2]

            # make copy of reference/gt
            shutil.copy(reference_path, f"{eval_dir}/{save_dirname}/ground_truth.{output_extension}")
            
            
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
            loss_fn = mean_relative_l2
        else:
            # SDF data
            assert args.mesh
            sdf_dataset = SDFDataset(reference_path, device=device, clip_sdf=None, num_samples=lora_config["batch_size"], transformation_save_dir=base_model_checkpoint_dir)
            n_channels = 1
            spatial_input_dim = 3
            output_extension = "obj"
            loss_fn = mape_loss

            
        results = {}

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
        num_base_grid_params = sum(p.numel() for p in encoding.parameters())
        num_base_mlp_params = sum(p.numel() for p in mlp.parameters())

        # separate grid encoding and mlp
        encoding = model[0]
        for params in encoding.parameters():
            params.requires_grad = False
        encoding.to(device)

        mlp = model[1]

        if not lora_config["full_finetuning"]:
            print("USING LORA")
            # initialize and load trained lora (with previous loras if nonempty)
            mlp.initialize_lora(rank=lora_config["rank"], prev_frame_lora_paths=prev_frame_checkpoint_paths, device=device) # also freezes trained base model's weights
            mlp.lora_mlps.load_state_dict(torch.load(f"{args.lora_or_finetuned_checkpoint_dir}/{save_dirname}/lora_weights.pt", weights_only=True)['model_state_dict'])
            mlp.to(device)
            forward_fn = lambda input_pos : mlp.forward_with_lora(input_pos, encoding(input_pos))
            if train_video_lora:
                # store path to current weights (to be trained) for next frame
                prev_frame_checkpoint_paths.append(f"{args.lora_or_finetuned_checkpoint_dir}/{save_dirname}/lora_weights.pt")
            results["rank"] = lora_config["rank"]
        else:
            # initialize and load fully finetuned model
            mlp.load_state_dict(torch.load(f"{args.lora_or_finetuned_checkpoint_dir}/{save_dirname}/finetuned_weights.pt", weights_only=True)['model_state_dict'])
            forward_fn = lambda input_pos : mlp(encoding(input_pos))
            prev_frame_checkpoint_paths.append(f"{args.lora_or_finetuned_checkpoint_dir}/{save_dirname}/finetuned_weights.pt")
            results["rank"] = "FULL FINETUNING"
        mlp.to(device)

        print(f"Number of base model parameters (grid): {num_base_grid_params}")
        print(f"Number of base model parameters (mlp): {num_base_mlp_params}")
        trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
        print(f"Trainable (LoRA or finetuneable) parameters: {trainable_params}")




        # reconstruct from trained neural field and compute metrics against reference 
        os.makedirs(f"{eval_dir}/{save_dirname}", exist_ok=True)
        save_path = f"{eval_dir}/{save_dirname}/best_reconstruction.{output_extension}"
        print(f"Saving reconstruction at:   {save_path}  ...")
        results['command'] = 'python '+ ' '.join(sys.argv)
        results['num_parameters'] = trainable_params
        if args.image or args.video:
            reconstruction = forward_fn(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
            # save best reconstructed image
            with torch.no_grad():
                write_image(save_path, reconstruction)
            reconstructed_image = Image(save_path, device) 
            # save difference image vs GT
            difference_image = np.abs(reference_image.orig_data_npy[:,:,:-1] - reconstructed_image.orig_data_npy[:,:,:-1]) # ignore alpha channel
            with torch.no_grad():
                # save raw difference as np array
                np.save(f"{eval_dir}/{save_dirname}/difference.npy", difference_image) # shape [H,W,3]
                write_image(f"{eval_dir}/{save_dirname}/difference.{output_extension}", difference_image)

            # grayscale difference
            grayscale_difference = np.mean(difference_image, axis=-1)
            grayscale_difference_3channel = np.stack([grayscale_difference, grayscale_difference, grayscale_difference], axis=-1)
            with torch.no_grad():
                write_image(f"{eval_dir}/{save_dirname}/difference_grayscale.{output_extension}", grayscale_difference_3channel)

            # invert grayscale difference image (high error = black)
            difference_image_inverted = 1. - grayscale_difference_3channel
            with torch.no_grad():
                write_image(f"{eval_dir}/{save_dirname}/difference_inverted.{output_extension}", difference_image_inverted)

            # compute PSNR vs. reference
            psnr = peak_signal_noise_ratio(reference_image.orig_data_npy, reconstructed_image.orig_data_npy)
            mse = mean_squared_error(reference_image.orig_data_npy, reconstructed_image.orig_data_npy)
            training_loss = mean_relative_l2(torch.tensor(reconstructed_image.orig_data_npy), torch.tensor(reference_image.orig_data_npy)).item()
            print(f"PSNR:  {psnr}")
            print(f"MSE:  {mse}")
            print(f"mean_relative_l2 (train loss):  {training_loss}")
            # with open(f"{eval_dir}/{save_dirname}/psnr.txt", "w") as file:
            #     file.write(str(psnr))
            results['psnr'] = psnr
            results['mse'] = mse
            results['training_loss'] = training_loss
            
            if args.video:
                psnr_per_frame.append(psnr) 
        else:
            assert args.mesh
            save_mesh(save_path, forward_fn, device, resolution=512*2)
            print("<=== done.")
            ## compute IoU using sign of SDF samples in uniform grid 
            # sample query points uniformly in [-1,1]^3
            grid_resolution = 512
            model_sdf_vals = extract_fields(bound_min=torch.FloatTensor([-1, -1, -1]), 
                                            bound_max=torch.FloatTensor([1, 1, 1]), 
                                            resolution=grid_resolution, 
                                            query_func=forward_fn,
                                            device=device)   # [B,]
            model_sdf_vals = torch.tensor(model_sdf_vals, device=device)
            model_interior_points = (model_sdf_vals < 0) # bools [B,]
 
            def reference_query_func(pts): # torch wrapper of numpy SDF func of mesh
                pts = pts.detach().cpu().numpy()
                return torch.tensor(-sdf_dataset.sdf_fn(pts)[:,None].astype(np.float32)).to(device)
            
            reference_sdf_vals = extract_fields(bound_min=torch.FloatTensor([-1, -1, -1]), 
                                            bound_max=torch.FloatTensor([1, 1, 1]), 
                                            resolution=grid_resolution, 
                                            query_func=reference_query_func)   # [B,]
            reference_sdf_vals = torch.tensor(reference_sdf_vals, device=device)
            reference_interior_points = (reference_sdf_vals < 0) # bools  [B,]

            # print(torch.sum(model_interior_points))
            # print(torch.sum(reference_interior_points))

            IoU = torch.sum(model_interior_points & reference_interior_points)/torch.sum(model_interior_points | reference_interior_points) #  == (num intersection samples / num union samples)
            print(f"IoU:  {IoU}")
            results['iou'] = IoU.item()
            # with open(f"{eval_dir}/{save_dirname}/iou.txt", "w") as file:
            #     file.write(str(IoU.item()))

        # save results json
        with open(f"{eval_dir}/{save_dirname}/results.json", "w") as outfile:
            json.dump(results, outfile, indent=4)

    if args.video:
        print("="*30)
        # collect and save all frames' psnr 
        for i, psnr in enumerate(psnr_per_frame):
            print(f"frame {i+1}: {psnr}")
        with open(f"{eval_dir}/all_frames_psnr.txt", "w") as file:
            string_psnrs = [str(psnr) for psnr in psnr_per_frame]
            file.write('\n'.join(string_psnrs))

        mean_psnr = np.mean(psnr_per_frame)
        median_psnr = np.median(psnr_per_frame)
        min_psnr = np.amin(psnr_per_frame)
        max_psnr = np.amax(psnr_per_frame)

        print("mean PSNR: ", mean_psnr)
        print("median PSNR: ", median_psnr)
        print("min PSNR: ", min_psnr)
        print("max PSNR: ", max_psnr)
        with open(f"{eval_dir}/mean_psnr.txt", "w") as file:
            file.write(str(mean_psnr))
        with open(f"{eval_dir}/median_psnr.txt", "w") as file:
            file.write(str(median_psnr))
        with open(f"{eval_dir}/min_psnr.txt", "w") as file:
            file.write(str(min_psnr))
        with open(f"{eval_dir}/max_psnr.txt", "w") as file:
            file.write(str(max_psnr))
        