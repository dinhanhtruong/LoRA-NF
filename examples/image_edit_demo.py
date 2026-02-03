import torch
import torch.nn as nn
from lora_nf.torch_modules import CustomFrequencyEncoding
from lora_nf.util import mean_relative_l2, get_device
from lora_nf.data_samplers import Image
from lora_nf.train_lora import train_base_model, train_lora_regression


def image_demo():
    # demo for using LoRA to encode an image edit
    save_dir = "checkpoints/minimal"
    base_image_path= "data/images/table/table_before.png"
    target_image_path= "data/images/table/table_after.png"
    device = torch.device("cuda:0")

    # set up base model
    pos_enc = CustomFrequencyEncoding()
    base_nf = nn.Sequential(
        pos_enc,
        nn.Linear(pos_enc.get_encoding_output_dim(2), 128), 
        nn.ReLU(),        
        nn.Linear(128, 128), 
        nn.ReLU(),         
        nn.Linear(128, 128), 
        nn.ReLU(),         
        nn.Linear(128, 128), 
        nn.ReLU(),         
        nn.Linear(128, 128), 
        nn.ReLU(),         
        nn.Linear(128, 3), # RGB output 
    )
    base_nf.to(device)
    base_data_sampler = Image(base_image_path, device)
    loss_fn = mean_relative_l2 
    base_nf = train_base_model(base_nf, base_data_sampler, loss_fn, save_dir=save_dir, max_n_steps=100000)
    
    #########################
    # train lora (outputs are saved in checkpoints/minimal/)
    target_data_sampler = Image(target_image_path, get_device(base_nf)) 
    lora_rank = 16

    lora_weights, lora_nf = train_lora_regression(base_nf, target_data_sampler, loss_fn, lora_rank, save_dir=save_dir, max_n_steps=30000)
    
    print("Done!")

if __name__ == "__main__":
    # this should be run from the project root
    print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")
    torch.set_printoptions(precision=7)
    try: 
        print("Running image demo...")
        image_demo()
    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Exiting...")