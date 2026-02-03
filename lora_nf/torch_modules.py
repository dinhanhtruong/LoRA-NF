import math
import torch
import torch.nn as nn

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

    def forward(self, x):
        return self.sequential(x)
    

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