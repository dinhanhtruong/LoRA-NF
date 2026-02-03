import itertools
import math
import sys
import numpy as np
import torch
import torch.nn as nn
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

class CustomFrequencyEncoding(nn.Module):
    def __init__(self):
        super(CustomFrequencyEncoding, self).__init__()
    def get_encoding_output_dim(self, input_dim):
        return self.forward(torch.zeros((1, input_dim))).shape[-1]

    def forward( # from https://github.com/krrish94/nerf-pytorch/blob/master/nerf/nerf_helpers.py
        self, tensor, num_encoding_functions=10, include_input=True, log_sampling=True
    ) -> torch.Tensor:
        r"""Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """
        encoding = [tensor] if include_input else []
        frequency_bands = None
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (num_encoding_functions - 1),
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim, hidden_activation_fn, output_activation_fn):
        super(MLP, self).__init__()

        # construct MLP: R^input_dim --> R^output_dim
        layers = [] 
        if num_hidden_layers == 0:
            layers.append(nn.Linear(input_dim, output_dim)) # single input->output layer
        else:
            layers.append(nn.Linear(input_dim, hidden_dim)) # initial layer
            layers.append(hidden_activation_fn)
            for _ in range(num_hidden_layers): # hidden layers
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(hidden_activation_fn)
            layers.append(nn.Linear(hidden_dim, output_dim)) # output layer
        if output_activation_fn is not None:
            layers.append(output_activation_fn)

        # stack layers sequantially 
        # (note: weights/biases can be extracted after initialization using model_name.layers[i].weight or model_name.layers[i].bias)
        self.layers = nn.Sequential(*layers)

        print(self.layers)
        self.use_lora = False

    def get_num_linear_layers(self):
        num_linear = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                num_linear += 1
        return num_linear

    def initialize_lora(self, rank, edit_region=[], prev_frame_lora_paths=[], device=None):
        """
        Initializes LoRA MLP per subregion of the input. 
        NOTE: assumes that this parent MLP model has been properly initialized (with weights loaded)

        prev_frame_lora_paths: list of chronologically ordered filepaths to prev frames' LoraMLP weights 
        """
        print("initializing lora MLPs")
        self.use_lora = True

        # freeze pre-trained mlp weights
        for params in self.parameters():
            params.requires_grad = False

        if edit_region:
            self.regions = [edit_region]
            print(f"{len(self.regions)} regions: {self.regions}")
        
        print(f"base model num linear layers: {self.get_num_linear_layers()}")


        # initialize LoRA grid: one lora per grid point 
        self.lora_mlps = nn.ModuleList([
            LoRA_MLP(self, 
                    0, 
                    rank, 
                    prev_frame_lora_paths=prev_frame_lora_paths, device=device)])

        ### TEMP: train single MLP from scratch in edit region
        # print("TRAINING LOCAL MLP FROM SCRATCH")
        # self.lora_mlps = nn.ModuleList([
        #     MLP( 
        #         input_dim=2, # 2D position on image
        #         output_dim=1, # intensity value
        #         num_hidden_layers=1,
        #         hidden_dim=26,
        #         hidden_activation_fn=nn.ReLU(),
        #         output_activation_fn=nn.Sigmoid(),
        #         use_positional_encoding=True
        #     ) for _ in range(len(self.regions)) 
        # ])
        # print("LoRA MLP structure: \n", self.lora_mlps)
        ###

    def get_mean_abs_lora_weight(self, num_lora_regions=4):
        print("Mean lora weight magnitudes")
        mean_weights = torch.zeros(num_lora_regions)
        for region_index, lora_mlp in enumerate(self.lora_mlps):
            mean_weights[region_index] = lora_mlp.get_mean_abs_lora_weight()
            print(f"\t\t region {region_index}:  {mean_weights[region_index]}")
        
        mean_weight = torch.mean(mean_weights).item()
        print("\t mean weight: ", mean_weight)
        return mean_weight


    def forward(self, x):
        """
        input: [B, input_dim]
        x: input spatial positions in [0,1]^2 where origin (0,0) is at top left corner of image and (1,1) is bottom right. [B, 2]
        edit_region_only: if true, assume that x is only in edit region.
        """
        assert not self.use_lora
        return self.layers(x)
    
    def forward_with_lora(self, x, input_feats):
        """
        x: input spatial positions in [0,1]^2 where origin (0,0) is at top left corner of image and (1,1) is bottom right. [B, 2] for images
        input_feats: [B, input_dim]
        """
        assert self.use_lora
        return self.lora_mlps[0](input_feats, self, use_pos_enc=False, lora_weight=1)
        

    def custom_weight_lora_linear_forward(self, x, base_linear, lora_matrix, lora_scaling):
        """
        x: [B, layer_in]
        lora_matrix: batched lora AB matrix of size [B, layer_in, layer_out]
        
        Returns [B, layer_out]
        """
        return  base_linear(x) + torch.bmm(x.unsqueeze(1), lora_matrix).squeeze(1) *lora_scaling
        
    
# class LoRA_MLP(nn.Module):
#     def __init__(self, parent_mlp, region_index, rank, num_active_layers_from_end="all", shared_A_matrices=None, prev_frame_lora_paths=[], device=None):
#         '''
#         LoRA-augmented MLP responsible for a sub-region of the parent's spatial range 

#         prev_frame_lora_paths: list of chronologically ordered filepaths to prev frames' LoraMLP weights 
#         '''
#         super(LoRA_MLP, self).__init__()
#         self.region_index = region_index
#         if prev_frame_lora_paths:
#             prev_frame_loras = [torch.load(path, weights_only=True) for path in prev_frame_lora_paths] # list of LoRA_MLP state dicts

#         # for each parent MLP linear layer, store a low-rank adaptor (LoRALinear layer).
#         # mimic parent MLP but replace linear with LoRALinear
#         self.lora_layers =  nn.ModuleDict()
#         assert num_active_layers_from_end == "all" or num_active_layers_from_end > 0
#         first_lora_layer_idx = 0 
#         if num_active_layers_from_end != "all":
#             first_lora_layer_idx = parent_mlp.get_num_linear_layers() - num_active_layers_from_end

#         linear_layer_idx = 0
#         for global_layer_idx, parent_layer in enumerate(parent_mlp.layers):
#             if isinstance(parent_layer, nn.Linear):
#                 # add lora layer
#                 if linear_layer_idx >= first_lora_layer_idx:
#                     A = None if shared_A_matrices is None else shared_A_matrices[linear_layer_idx]
#                     prev_frame_lora_matrices = []
#                     if prev_frame_lora_paths:
#                         for prev_frame_lora in prev_frame_loras:
#                             curr_A = prev_frame_lora["model_state_dict"][f"0.lora_layers.{global_layer_idx}.A"]
#                             curr_B = prev_frame_lora["model_state_dict"][f"0.lora_layers.{global_layer_idx}.B"]
#                             if device is not None:
#                                 curr_A = curr_A.to(device)
#                                 curr_B = curr_B.to(device)
#                             prev_frame_lora_matrices.append(curr_A @ curr_B)
#                     self.lora_layers[str(global_layer_idx)] = LoRALinear(parent_layer, r=rank, A=A, prev_frame_lora_matrices=prev_frame_lora_matrices)
#                 linear_layer_idx += 1

#         # print("lora mlp: \n", self.layers)

#     def get_mean_abs_lora_weight(self):
#         # for each linear layer:
#             # store mean(abs(W))
#         # average at end
#         mean_abs_weights = []
#         for layer in self.lora_layers.values():
#             assert isinstance(layer, LoRALinear)
#             delta_w = layer.A @ layer.B
#             mean_abs_weights.append(torch.mean(torch.abs(delta_w)))
#         return torch.mean(torch.tensor(mean_abs_weights))


#     def forward(self, x, parent_mlp, use_pos_enc, lora_weight=1):
#         for i, parent_layer in enumerate(parent_mlp.layers):
#             if str(i) in self.lora_layers:
#                 # apply lora linear forward (this linear layer is active)
#                 x = self.lora_layers[str(i)](x, parent_layer, lora_weight)
#             else:
#                 # apply base mlp layer (non-linearities or lora-inactive layers) 
#                 x = parent_layer(x)
#         return x



########################

class LoRA_MLP(nn.Module):
    def __init__(self, base_mlp, rank):
        '''
        LoRA-augmented MLP 

        prev_frame_lora_paths: list of chronologically ordered filepaths to prev frames' LoraMLP weights 
        '''
        super(LoRA_MLP, self).__init__()

        # for each parent MLP linear layer, store a low-rank adaptor (LoRALinear layer).
        # mimic parent MLP but replace linear with LoRALinear
        self.sequential = nn.Sequential() # just LoRALinears

        for base_layer in base_mlp:
            if isinstance(base_layer, nn.Linear):
                # add lora layer
                self.sequential.append(LoRALinear(base_layer, r=rank))
            else:
                # keep parent layer (activation or positional encoding)
                self.sequential.append(base_layer)
  
    def as_sequential(self):
        return self.sequential
    def get_lora_weights(self):
        """
        Returns a list of tensors of weights corresponding to the current LoRAs (in the same order as the base model's)
        """
        lora_weights = []
        for lora_mlp_layer in self.sequential:
            if isinstance(lora_mlp_layer, LoRALinear):
                lora_weights.append((lora_mlp_layer.A @ lora_mlp_layer.B).T) # [in_features, out_features]
        return lora_weights


    # def get_mean_abs_lora_weight(self):
    #     # for each linear layer:
    #         # store mean(abs(W))
    #     # average at end
    #     mean_abs_weights = []
    #     for layer in self.sequential.values():
    #         assert isinstance(layer, LoRALinear)
    #         delta_w = layer.A @ layer.B
    #         mean_abs_weights.append(torch.mean(torch.abs(delta_w)))
    #     return torch.mean(torch.tensor(mean_abs_weights))


    def forward(self, x):
        return self.sequential(x)

        for i, parent_layer in enumerate(parent_mlp.layers):
            if str(i) in self.sequential:
                # apply lora linear forward (this linear layer is active)
                x = self.sequential[str(i)](x, parent_layer, lora_weight)
            else:
                # apply base mlp layer (non-linearities or lora-inactive layers) 
                x = parent_layer(x)
        return x
    

class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, alpha=1, A=None):
        '''
        
        '''
        super(LoRALinear, self).__init__()
        assert isinstance(base_linear, nn.Linear) and r > 1
        self.base_linear = base_linear
        # freeze base; make sure only lora weights are trainable
        self.base_linear.weight.requires_grad_(False)
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad_(False)

        rank = min(r, base_linear.in_features, base_linear.out_features)
        if A is not None:
        # use specified A matrix
            assert A.shape == torch.Size((base_linear.in_features, rank))
            self.A = A
        else:
            self.A = nn.Parameter(torch.empty((base_linear.in_features, rank)))
            nn.init.normal_(self.A, std=1/math.sqrt(base_linear.in_features)) #following pg4 of https://arxiv.org/pdf/2406.08447 
        self.B = nn.Parameter(torch.zeros(rank, base_linear.out_features))
        # lora scale factor
        self.scaling = alpha/r

    def forward(self, x):
        return self.base_linear(x) + torch.linalg.multi_dot((x, self.A, self.B))*self.scaling
    

def extract_linear_layers(module):
    return [m for m in module.modules() if isinstance(m, nn.Linear)]