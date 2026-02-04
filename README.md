# LoRA-NF
This small python package implements the method from [Low-Rank Adaptation of Neural Fields](https://dl.acm.org/doi/10.1145/3757377.3763882).

In particular, it provides a simple function ```train_lora_regression()``` for adapting any MLP-based neural field (implemented as a ```torch.nn.Sequential()```) to new data using low-rank adaptations (LoRA). 

## Installation
Install the required dependencies in a conda environment using the provided ```environment.yml``` by running
```
conda env create -f environment.yml
```
Then install the ```lora-nf``` package:
```
pip install lora-nf
```

## Simple Example
The code snippet below (see ```examples/image_edit_demo.py```) shows how this library can be used to (1) train a base image/RGB neural field and (2) fine-tune the base neural field to regress an edited image.

```python
import torch
import torch.nn as nn
from lora_nf.torch_modules import CustomFrequencyEncoding
from lora_nf.util import mean_relative_l2, get_device
from lora_nf.data_samplers import Image
from lora_nf.train_lora import train_base_model, train_lora_regression

## demo for using LoRA to encode an image edit
save_dir = "checkpoints/minimal"
base_image_path= "data/images/table/table_before.png"
target_image_path= "data/images/table/table_after.png"
device = torch.device("cuda:0")

# train base neural field
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

# train LoRA (outputs are saved in checkpoints/minimal/)
target_data_sampler = Image(target_image_path, get_device(base_nf)) 
lora_rank = 16

lora_weights, lora_nf = train_lora_regression(base_nf, target_data_sampler, loss_fn, lora_rank, save_dir=save_dir, max_n_steps=30000)
```

You can run the example above using 
```
python examples/image_edit_demo.py
```  
