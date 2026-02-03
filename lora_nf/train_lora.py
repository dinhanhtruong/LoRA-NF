import os
import random
import numpy as np
import time
import torch
import copy
import torch.nn as nn
from .torch_modules import LoRA_MLP, extract_linear_layers
from .util import get_device
from .data_samplers import DataSampler
from typing import Callable

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


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
    device = get_device(base_nf)
    linear_layers = extract_linear_layers(base_nf)
    base_output_dim = linear_layers[-1].out_features
    _, dummy_targets = target_sampler.sample_batch(batch_size, device)
    assert dummy_targets.shape[-1] == base_output_dim, "base_nf and target_sampler are incompatible: output size mismatch"
    assert isinstance(base_nf, nn.Sequential)
    
    # set up lora neural field
    lora_nf = LoRA_MLP(base_nf, lora_rank)
    lora_nf.to(device)
    print(f"training LoRA on {device}")
    trainable_params = sum(p.numel() for p in lora_nf.parameters() if p.requires_grad)
    print(f"# trainable LoRA parameters: {trainable_params}")

    optimizer = torch.optim.Adam(lora_nf.parameters(), lr=learning_rate) # only fine-tune mlp
    # use learning rate scheduler if specified
    scheduler = None        
    if lr_scheduler_warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=lr_scheduler_warmup_steps)
    # for logging
    best_loss = float('inf')
    best_lora_nf = copy.deepcopy(lora_nf)
    prev_time = time.perf_counter()
    periods_no_improve = 0 # for early stopping
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(max_n_steps):
        batch_inputs, batch_targets = target_sampler.sample_batch(batch_size, device) # [B,in], [B,out]
        output = lora_nf(batch_inputs.to(device))
        loss = loss_fn(output, batch_targets.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i % log_interval == 0: # reconstruct output
            curr_loss = loss.item()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={curr_loss:.7f} time={int(elapsed_time)}[s]")
            if save_dir:
                with torch.no_grad():
                    target_sampler.save_model_output(lora_nf, save_path=f"{save_dir}/lora_nf_step_{i:05d}")

            if curr_loss < best_loss:
                print(f"\tdecreased {best_loss:.6f}-->{curr_loss:.6f}")
                best_loss = curr_loss
                periods_no_improve = 0 # reset
                best_lora_nf = copy.deepcopy(lora_nf)
            else:
                # early stopping if no improvement for several epochs
                periods_no_improve += 1
                if periods_no_improve >= convergence_patience:
                    print(f"Early stopping at step {i} with loss {curr_loss}")
                    break
            prev_time = time.perf_counter()
            if i > 0 and log_interval < 1000: 
                log_interval *= 10
    if save_dir:
        torch.save({
            'step': i,
            'model_state_dict': best_lora_nf.state_dict(),
            'loss': curr_loss,
        }, f"{save_dir}/lora_nf_best.pt")
        target_sampler.save_model_output(best_lora_nf, save_path=f"{save_dir}/lora_nf_best")
        
    return best_lora_nf.get_lora_weights(), best_lora_nf.as_sequential()

def train_base_model(
        base_nf: nn.Sequential, 
        data_sampler: DataSampler, 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        learning_rate=1e-4, 
        batch_size=2**18, 
        max_n_steps=100000, 
        lr_scheduler_warmup_steps=0, 
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
        batch_inputs, batch_targets = data_sampler.sample_batch(batch_size, device) # [B,in], [B,out]
        output = base_nf(batch_inputs.to(device))
        loss = loss_fn(output, batch_targets.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i % log_interval == 0: # reconstruct output
            curr_loss = loss.item()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={curr_loss:.7f} time={int(elapsed_time)}[s]")
            if save_dir:
                with torch.no_grad():
                    data_sampler.save_model_output(base_nf, save_path=f"{save_dir}/base_nf_step_{i:05d}")

            if curr_loss < best_loss:
                print(f"\tdecreased {best_loss:.6f}-->{curr_loss:.6f}")
                best_loss = curr_loss
                periods_no_improve = 0 # reset
                best_model = copy.deepcopy(base_nf)
            else:
                # early stopping if no improvement for several epochs
                periods_no_improve += 1
                if periods_no_improve >= convergence_patience:
                    print(f"Early stopping at step {i} with loss {curr_loss}")
                    break
            prev_time = time.perf_counter()
            if i > 0 and log_interval < 1000: 
                log_interval *= 10
    if save_dir:
        data_sampler.save_model_output(best_model, save_path=f"{save_dir}/base_nf_best")
        torch.save({
            'step': i,
            'model_state_dict': best_model.state_dict(),
            'loss': curr_loss,
        }, f"{save_dir}/base_nf_best.pt")

    return best_model

