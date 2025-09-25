#!/usr/bin/env python3
"""
Phase 2.2: Genomic Transformer Encoder for GIMAN

Implements a transformer-based encoder for genetic variant modeling with multi-head
self-attention for gene-gene interactions. Processes LRRK2, GBA, and APOE_RISK 
genetic features from PPMI enhanced dataset.

Architecture:
- Genetic Feature Embedding: Maps genetic variants to high-dimensional space
- Position Embeddings: Encodes chromosomal locations and gene relationships  
- Multi-Head Self-Attention: Models gene-gene interactions and epistasis
- Layer Normalization: Stabilizes training for genetic data
- Output: 256-dimensional genetic embeddings per patient

Author: GIMAN Development Team
Date: September 24, 2025
Phase: 2.2 Genomic Transformer Encoder
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

class GenomicDataset(Dataset):
    """Dataset for genomic transformer encoder training."""
    
    def __init__(self, genetic_data: pd.DataFrame):
        """
        Initialize genomic dataset.
        
        Args:
            genetic_data: DataFrame with PATNO and genetic features
        """
        self.data = genetic_data.copy()
        self.patient_ids = self.data['PATNO'].values
        
        # Genetic feature columns
        self.genetic_features = ['LRRK2', 'GBA', 'APOE_RISK']
        
        # Extract genetic feature matrix
        self.genetic_matrix = self.data[self.genetic_features].values.astype(np.float32)
        
        # Gene position embeddings (approximate chromosomal positions)
        self.gene_positions = {
            'LRRK2': 0,   # Chromosome 12
            'GBA': 1,     # Chromosome 1  
            'APOE_RISK': 2  # Chromosome 19
        }
        
        # Create position encoding for each gene
        self.position_ids = torch.tensor([
            self.gene_positions['LRRK2'],
            self.gene_positions['GBA'], 
            self.gene_positions['APOE_RISK']
        ], dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get genomic features for a patient."""
        return {
            'patient_id': self.patient_ids[idx],
            'genetic_features': torch.tensor(self.genetic_matrix[idx], dtype=torch.float32),
            'position_ids': self.position_ids.clone()
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for genomic locations."""
    
    def __init__(self, d_model: int, max_genes: int = 100):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_genes, d_model)
        position = torch.arange(0, max_genes, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            position_ids: [batch_size, seq_len] or [seq_len]
            
        Returns:
            Positional encodings [batch_size, seq_len, d_model]
        """
        if position_ids.dim() == 1:
            # Single sequence
            return self.pe[position_ids]
        else:
            # Batch of sequences
            batch_size, seq_len = position_ids.shape
            pos_encodings = []
            for i in range(batch_size):
                pos_encodings.append(self.pe[position_ids[i]])
            return torch.stack(pos_encodings)

class MultiHeadGeneAttention(nn.Module):
    """Multi-head self-attention for gene-gene interactions."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        residual = x
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        output = self.w_o(attention_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output

class GenomicTransformerEncoder(nn.Module):
    """Transformer encoder for genomic variant modeling."""
    
    def __init__(
        self,
        n_genetic_features: int = 3,
        d_model: int = 256,
        n_heads: int = 8, 
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.n_genetic_features = n_genetic_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input embedding for genetic features
        self.genetic_embedding = nn.Linear(1, d_model)  # Each gene feature -> d_model
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer layers
        self.attention_layers = nn.ModuleList([
            MultiHeadGeneAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * n_genetic_features, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, genetic_features: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through genomic transformer.
        
        Args:
            genetic_features: [batch_size, n_genes] genetic variant values
            position_ids: [batch_size, n_genes] or [n_genes] position indices
            
        Returns:
            Genetic embeddings [batch_size, output_dim]
        """
        batch_size, n_genes = genetic_features.shape
        
        # Embed each genetic feature independently
        # Reshape to [batch_size, n_genes, 1] for embedding
        genetic_features_expanded = genetic_features.unsqueeze(-1)
        embedded = self.genetic_embedding(genetic_features_expanded)  # [batch_size, n_genes, d_model]
        
        # Add positional encoding
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        pos_encodings = self.pos_encoding(position_ids)  # [batch_size, n_genes, d_model]
        x = embedded + pos_encodings
        x = self.dropout(x)
        
        # Apply transformer layers
        for i in range(self.n_layers):
            # Multi-head attention
            attention_output = self.attention_layers[i](x)
            
            # Feed-forward network with residual connection
            ff_input = attention_output
            ff_output = self.feed_forwards[i](ff_input)
            x = self.layer_norms[i](ff_output + ff_input)
        
        # Global pooling: flatten and project to final dimensions
        x_flattened = x.view(batch_size, -1)  # [batch_size, n_genes * d_model]
        genetic_embedding = self.output_projection(x_flattened)  # [batch_size, output_dim]
        
        return genetic_embedding

def create_genomic_dataset(enhanced_data_path: str) -> Tuple[GenomicDataset, Dict[str, Any]]:
    """
    Create genomic dataset from enhanced PPMI data.
    
    Args:
        enhanced_data_path: Path to enhanced dataset CSV
        
    Returns:
        Tuple of (dataset, metadata)
    """
    logger.info(f"Loading genetic data from {enhanced_data_path}")
    
    # Load enhanced dataset
    df = pd.read_csv(enhanced_data_path)
    logger.info(f"Loaded dataset with {len(df)} patients and {len(df.columns)} features")
    
    # Ensure genetic features are present
    genetic_features = ['LRRK2', 'GBA', 'APOE_RISK']
    missing_features = [f for f in genetic_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing genetic features: {missing_features}")
    
    # Create dataset
    dataset = GenomicDataset(df)
    
    # Create metadata
    metadata = {
        'n_patients': len(df),
        'genetic_features': genetic_features,
        'n_genetic_features': len(genetic_features),
        'feature_statistics': {}
    }
    
    # Add feature statistics
    for feature in genetic_features:
        stats = {
            'mean': float(df[feature].mean()),
            'std': float(df[feature].std()),
            'min': float(df[feature].min()),
            'max': float(df[feature].max()),
            'unique_values': sorted(df[feature].unique().tolist())
        }
        metadata['feature_statistics'][feature] = stats
    
    return dataset, metadata

def train_genomic_encoder(
    dataset: GenomicDataset,
    model: GenomicTransformerEncoder,
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Train genomic transformer encoder using contrastive learning.
    
    Args:
        dataset: Genomic dataset
        model: Genomic transformer model
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device
        
    Returns:
        Training results dictionary
    """
    logger.info(f"Training genomic encoder for {n_epochs} epochs")
    
    model = model.to(device)
    model.train()
    
    # Data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training loop
    training_losses = []
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        for batch in dataloader:
            genetic_features = batch['genetic_features'].to(device)
            position_ids = batch['position_ids'].to(device)
            
            # Forward pass
            embeddings = model(genetic_features, position_ids)
            
            # Contrastive loss: encourage diverse genetic embeddings
            # Normalize embeddings
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
            
            # Create labels (diagonal should be 1, off-diagonal should be low)
            batch_size = embeddings.shape[0]
            labels = torch.eye(batch_size, device=device)
            
            # Contrastive loss: maximize diagonal, minimize off-diagonal
            loss = F.mse_loss(similarity_matrix, labels)
            
            # Add diversity regularization
            mean_embedding = embeddings.mean(dim=0)
            diversity_loss = -torch.norm(embeddings - mean_embedding.unsqueeze(0), dim=1).mean()
            
            total_loss = loss + 0.1 * diversity_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_losses.append(total_loss.item())
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        training_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{n_epochs}: Loss = {avg_loss:.6f}")
    
    return {
        'training_losses': training_losses,
        'final_loss': training_losses[-1] if training_losses else 0.0,
        'n_epochs': n_epochs,
        'n_parameters': sum(p.numel() for p in model.parameters())
    }

def evaluate_genomic_encoder(
    model: GenomicTransformerEncoder,
    dataset: GenomicDataset,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Evaluate genomic encoder and analyze learned embeddings.
    
    Args:
        model: Trained genomic encoder
        dataset: Genomic dataset
        device: Evaluation device
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating genomic encoder")
    
    model = model.to(device)
    model.eval()
    
    # Generate embeddings for all patients
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_embeddings = []
    all_patient_ids = []
    all_genetic_features = []
    
    with torch.no_grad():
        for batch in dataloader:
            genetic_features = batch['genetic_features'].to(device)
            position_ids = batch['position_ids'].to(device)
            patient_ids = batch['patient_id'].cpu().numpy()
            
            embeddings = model(genetic_features, position_ids)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_patient_ids.append(patient_ids)
            all_genetic_features.append(genetic_features.cpu().numpy())
    
    # Concatenate results
    embeddings = np.vstack(all_embeddings)
    patient_ids = np.concatenate(all_patient_ids)
    genetic_features = np.vstack(all_genetic_features)
    
    # Analyze embeddings
    embedding_stats = {
        'mean': float(embeddings.mean()),
        'std': float(embeddings.std()),
        'l2_norm_mean': float(np.linalg.norm(embeddings, axis=1).mean()),
        'l2_norm_std': float(np.linalg.norm(embeddings, axis=1).std())
    }
    
    # Compute pairwise similarities
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    
    # Get upper triangle (excluding diagonal)
    n = similarity_matrix.shape[0]
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    pairwise_similarities = similarity_matrix[mask]
    
    diversity_stats = {
        'mean_pairwise_similarity': float(pairwise_similarities.mean()),
        'similarity_std': float(pairwise_similarities.std()),
        'min_similarity': float(pairwise_similarities.min()),
        'max_similarity': float(pairwise_similarities.max())
    }
    
    return {
        'embeddings': embeddings,
        'patient_ids': patient_ids,
        'genetic_features': genetic_features,
        'embedding_stats': embedding_stats,
        'diversity_stats': diversity_stats,
        'n_patients': len(patient_ids),
        'embedding_dim': embeddings.shape[1]
    }

def main():
    """Main training and evaluation pipeline for Phase 2.2."""
    
    print("🧬 PHASE 2.2: GENOMIC TRANSFORMER ENCODER")
    print("=" * 60)
    
    # Configuration
    config = {
        'enhanced_data_path': 'data/enhanced/enhanced_dataset_latest.csv',
        'model_params': {
            'n_genetic_features': 3,
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 1024,
            'dropout': 0.1,
            'output_dim': 256
        },
        'training_params': {
            'n_epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create dataset
        print("\n📊 Creating genomic dataset...")
        dataset, metadata = create_genomic_dataset(config['enhanced_data_path'])
        print(f"✅ Dataset created: {len(dataset)} patients, {metadata['n_genetic_features']} genetic features")
        
        # Print genetic feature statistics
        print(f"\n🧬 Genetic feature statistics:")
        for feature, stats in metadata['feature_statistics'].items():
            print(f"  {feature}:")
            print(f"    Values: {stats['unique_values']}")
            print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        
        # Create model
        print(f"\n🏗️ Creating genomic transformer encoder...")
        model = GenomicTransformerEncoder(**config['model_params'])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {n_params:,} parameters")
        print(f"   Architecture: {config['model_params']['n_layers']} layers, {config['model_params']['n_heads']} heads")
        print(f"   Input: {config['model_params']['n_genetic_features']} genetic features")
        print(f"   Output: {config['model_params']['output_dim']}-dimensional embeddings")
        
        # Train model
        print(f"\n🚂 Training genomic encoder...")
        training_results = train_genomic_encoder(
            dataset, model, device=device, **config['training_params']
        )
        print(f"✅ Training completed: Final loss = {training_results['final_loss']:.6f}")
        
        # Evaluate model
        print(f"\n📈 Evaluating genomic encoder...")
        evaluation_results = evaluate_genomic_encoder(model, dataset, device)
        
        print(f"\n🎯 GENOMIC ENCODER RESULTS:")
        print(f"   Patients processed: {evaluation_results['n_patients']}")
        print(f"   Embedding dimensions: {evaluation_results['embedding_dim']}")
        print(f"   Mean L2 norm: {evaluation_results['embedding_stats']['l2_norm_mean']:.3f}")
        print(f"   Embedding diversity: {evaluation_results['diversity_stats']['mean_pairwise_similarity']:.6f}")
        
        # Interpret diversity
        diversity = evaluation_results['diversity_stats']['mean_pairwise_similarity']
        if diversity < 0.2:
            quality = "EXCELLENT"
            print(f"   ✅ {quality}: Highly diverse genetic representations!")
        elif diversity < 0.5:
            quality = "GOOD" 
            print(f"   ✅ {quality}: Well-separated genetic profiles")
        else:
            quality = "MODERATE"
            print(f"   ⚠️ {quality}: Some genetic similarity")
        
        # Save model and results
        output_dir = "models"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{output_dir}/genomic_transformer_encoder_phase2_2.pth"
        
        # Save comprehensive checkpoint
        checkpoint = {
            **{f'model_state_dict': model.state_dict()},
            'config': config,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'metadata': metadata,
            'timestamp': timestamp,
            'phase': '2.2_genomic_transformer'
        }
        
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print(f"✅ Model saved: {model_path}")
        
        print(f"\n🎉 PHASE 2.2 GENOMIC ENCODER COMPLETE!")
        print(f"   • Genetic transformer successfully trained on {evaluation_results['n_patients']} patients")
        print(f"   • {quality} genetic embeddings achieved (similarity: {diversity:.6f})")
        print(f"   • Ready for Phase 3: Graph-Attention Fusion with Phase 2.1 spatiotemporal encoder")
        
        return model, evaluation_results
        
    except Exception as e:
        logger.error(f"Phase 2.2 failed: {e}")
        raise

if __name__ == "__main__":
    model, results = main()