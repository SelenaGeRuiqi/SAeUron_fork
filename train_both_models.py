"""
Fair comparison training script with configurable parameters
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from tqdm import tqdm
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime

from models.traditional_sae import TraditionalSAE
from models.diffusion_msae import DiffusionMSAE

class ActivationDataset:
    """Simple dataset for our collected activations"""
    def __init__(self, activations_path):
        print(f"Loading activations from {activations_path}")
        with open(activations_path, 'rb') as f:
            self.activations = pickle.load(f)
        
        # Flatten the batch dimension
        self.samples = []
        for activation in self.activations:
            for i in range(activation.shape[0]):
                self.samples.append(activation[i])
        
        self.samples = torch.stack(self.samples).float()
        print(f"Dataset ready: {len(self.samples)} samples of shape {self.samples[0].shape}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def train_traditional_sae(dataset, config):
    """Train traditional SAE with configurable parameters"""
    print(f"\n=== Training Traditional SAE ===")
    print(f"Config: expansion_factor={config.expansion_factor}, k={config.k}")
    
    # Create model
    sae = TraditionalSAE(
        input_dim=1280,
        expansion_factor=config.expansion_factor,
        k=config.k
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = sae.to(device)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in sae.parameters())
    print(f"Traditional SAE parameters: {total_params:,}")
    
    # Setup training
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = optim.Adam(sae.parameters(), lr=config.lr)
    
    # Training loop
    sae.train()
    losses = []
    
    for epoch in range(config.epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"SAE Epoch {epoch+1}/{config.epochs}")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Forward pass
            encoded, reconstructed = sae(batch)
            loss = sae.compute_loss(batch, reconstructed)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"SAE Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save checkpoint
    os.makedirs("sae-ckpts/traditional_sae", exist_ok=True)
    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': {
            'input_dim': 1280,
            'expansion_factor': config.expansion_factor,
            'k': config.k
        },
        'losses': losses
    }, f"sae-ckpts/traditional_sae/checkpoint_exp{config.expansion_factor}.pth")
    
    print(f"✅ Traditional SAE training completed. Final loss: {losses[-1]:.4f}")
    return sae, losses

def train_msae(dataset, config):
    """Train MSAE with configurable parameters"""
    print(f"\n=== Training MSAE ===")
    print(f"Config: expansion_factor={config.expansion_factor}, k_values={config.k_values}")
    
    # Create model
    msae = DiffusionMSAE(
        input_dim=1280,
        expansion_factor=config.expansion_factor,
        k_values=config.k_values
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    msae = msae.to(device)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in msae.parameters())
    print(f"MSAE parameters: {total_params:,}")
    
    # Setup training
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = optim.Adam(msae.parameters(), lr=config.lr)
    
    # Training loop
    msae.train()
    losses = []
    
    for epoch in range(config.epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"MSAE Epoch {epoch+1}/{config.epochs}")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Forward pass
            features, reconstructions = msae(batch)
            loss = msae.compute_loss(batch, reconstructions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"MSAE Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save checkpoint
    os.makedirs("sae-ckpts/msae", exist_ok=True)
    torch.save({
        'model_state_dict': msae.state_dict(),
        'config': {
            'input_dim': 1280,
            'expansion_factor': config.expansion_factor,
            'k_values': config.k_values
        },
        'losses': losses
    }, f"sae-ckpts/msae/checkpoint_exp{config.expansion_factor}.pth")
    
    print(f"✅ MSAE training completed. Final loss: {losses[-1]:.4f}")
    return msae, losses

def setup_logging(log_dir="log/train_both_models_log"):
    """Setup logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Setup logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

def train_traditional_sae(dataset, config, log_file):
    """Enhanced training with logging"""
    logging.info("=== Training Traditional SAE ===")
    logging.info(f"Config: expansion_factor={config.expansion_factor}, k={config.sae_k}")
    
    # Create model
    sae = TraditionalSAE(
        input_dim=1280,
        expansion_factor=config.expansion_factor,
        k=config.sae_k  # Use configurable k
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = sae.to(device)
    
    # Calculate and log total parameters
    total_params = sum(p.numel() for p in sae.parameters())
    logging.info(f"Traditional SAE parameters: {total_params:,}")
    
    # Setup training
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = optim.Adam(sae.parameters(), lr=config.lr)
    
    # Training loop
    sae.train()
    losses = []
    
    for epoch in range(config.epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"SAE Epoch {epoch+1}/{config.epochs}")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Forward pass
            encoded, reconstructed = sae(batch)
            loss = sae.compute_loss(batch, reconstructed)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        logging.info(f"SAE Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save checkpoint with config info
    os.makedirs("sae-ckpts/traditional_sae", exist_ok=True)
    checkpoint_name = f"checkpoint_exp{config.expansion_factor}_k{config.sae_k}.pth"
    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': {
            'input_dim': 1280,
            'expansion_factor': config.expansion_factor,
            'k': config.sae_k
        },
        'losses': losses,
        'final_loss': losses[-1],
        'total_params': total_params
    }, f"sae-ckpts/traditional_sae/{checkpoint_name}")
    
    logging.info(f"✅ Traditional SAE training completed. Final loss: {losses[-1]:.4f}")
    logging.info(f"Checkpoint saved: sae-ckpts/traditional_sae/{checkpoint_name}")
    
    return sae, losses

def train_msae(dataset, config, log_file):
    """Enhanced MSAE training with logging"""
    logging.info("=== Training MSAE ===")
    logging.info(f"Config: expansion_factor={config.expansion_factor}, k_values={config.msae_k_values}")
    
    # Create model
    msae = DiffusionMSAE(
        input_dim=1280,
        expansion_factor=config.expansion_factor,
        k_values=config.msae_k_values  # Use configurable k_values
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    msae = msae.to(device)
    
    # Calculate and log total parameters
    total_params = sum(p.numel() for p in msae.parameters())
    logging.info(f"MSAE parameters: {total_params:,}")
    
    # Setup training
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = optim.Adam(msae.parameters(), lr=config.lr)
    
    # Training loop
    msae.train()
    losses = []
    
    for epoch in range(config.epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"MSAE Epoch {epoch+1}/{config.epochs}")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Forward pass
            features, reconstructions = msae(batch)
            loss = msae.compute_loss(batch, reconstructions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        logging.info(f"MSAE Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save checkpoint with config info
    os.makedirs("sae-ckpts/msae", exist_ok=True)
    k_str = "_".join(map(str, config.msae_k_values))
    checkpoint_name = f"checkpoint_exp{config.expansion_factor}_k{k_str}.pth"
    torch.save({
        'model_state_dict': msae.state_dict(),
        'config': {
            'input_dim': 1280,
            'expansion_factor': config.expansion_factor,
            'k_values': config.msae_k_values
        },
        'losses': losses,
        'final_loss': losses[-1],
        'total_params': total_params
    }, f"sae-ckpts/msae/{checkpoint_name}")
    
    logging.info(f"✅ MSAE training completed. Final loss: {losses[-1]:.4f}")
    logging.info(f"Checkpoint saved: sae-ckpts/msae/{checkpoint_name}")
    
    return msae, losses

def main():
    parser = argparse.ArgumentParser(description="Train SAE vs MSAE with configurable parameters")
    parser.add_argument("--expansion_factor", type=int, default=16, 
                       help="Expansion factor for both models (default: 16)")
    parser.add_argument("--sae_k", type=int, default=16,
                       help="TopK value for traditional SAE (default: 16)")
    parser.add_argument("--msae_k_values", type=int, nargs="+", default=[16, 32, 64],
                       help="K values for MSAE hierarchy (default: [16, 32, 64])")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=4e-4,
                       help="Learning rate (default: 4e-4)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    
    logging.info("=== Day 2: Enhanced Training with Logging ===")
    logging.info(f"Expansion factor: {args.expansion_factor}")
    logging.info(f"SAE k: {args.sae_k}")
    logging.info(f"MSAE k_values: {args.msae_k_values}")
    logging.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Load dataset
    dataset = ActivationDataset("activations_poc/activations_unet_up_blocks_1_attentions_2.pkl")
    logging.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Train both models
    sae, sae_losses = train_traditional_sae(dataset, args, log_file)
    msae, msae_losses = train_msae(dataset, args, log_file)
    
    # Enhanced comparison and logging
    logging.info("=== Training Comparison ===")
    logging.info(f"Traditional SAE final loss: {sae_losses[-1]:.4f}")
    logging.info(f"MSAE final loss: {msae_losses[-1]:.4f}")
    
    # Parameter count comparison
    sae_params = sum(p.numel() for p in sae.parameters())
    msae_params = sum(p.numel() for p in msae.parameters())
    logging.info(f"SAE parameters: {sae_params:,}")
    logging.info(f"MSAE parameters: {msae_params:,}")
    logging.info(f"Parameter ratio (MSAE/SAE): {msae_params/sae_params:.2f}")
    
    # Detailed evaluation
    logging.info("=== Detailed Evaluation ===")
    test_batch = dataset.samples[:8].cuda()  # Larger test batch
    
    sae.eval()
    msae.eval()
    
    with torch.no_grad():
        # Traditional SAE evaluation
        sae_encoded, sae_recon = sae(test_batch)
        sae_mse = nn.functional.mse_loss(sae_recon, test_batch)
        sae_cosine = nn.functional.cosine_similarity(
            sae_recon.flatten(1), test_batch.flatten(1)
        ).mean()
        sae_sparsity = (sae_encoded > 0).float().mean()
        
        # MSAE evaluation (analyze all granularity levels)
        msae_features, msae_recons = msae(test_batch)
        
        logging.info("SAE Results:")
        logging.info(f"  MSE: {sae_mse.item():.4f}")
        logging.info(f"  Cosine Similarity: {sae_cosine.item():.4f}")
        logging.info(f"  Sparsity: {sae_sparsity.item():.1%}")
        
        logging.info("MSAE Results (by granularity):")
        for i, (features, recon) in enumerate(zip(msae_features, msae_recons)):
            mse = nn.functional.mse_loss(recon, test_batch)
            cosine = nn.functional.cosine_similarity(
                recon.flatten(1), test_batch.flatten(1)
            ).mean()
            sparsity = (features > 0).float().mean()
            k_val = args.msae_k_values[i]
            
            logging.info(f"  Level {i} (k={k_val}): MSE={mse.item():.4f}, "
                        f"Cosine={cosine.item():.4f}, Sparsity={sparsity.item():.1%}")
    
    # Winner determination
    best_msae_mse = min([nn.functional.mse_loss(recon, test_batch).item() 
                         for recon in msae_recons])
    
    if best_msae_mse < sae_mse.item():
        logging.info("🏆 MSAE wins on reconstruction quality!")
    else:
        logging.info("🏆 Traditional SAE wins on reconstruction quality!")
    
    logging.info("✅ Training and evaluation completed!")
    logging.info(f"All results logged to: {log_file}")
    
    return sae, msae

if __name__ == "__main__":
    sae, msae = main()