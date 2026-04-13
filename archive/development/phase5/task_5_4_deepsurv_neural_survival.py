"""
Task 5.4: DeepSurv Neural Survival Model for Phenoconversion Prediction

This script implements a deep learning-based survival analysis model (DeepSurv)
to predict phenoconversion from prodromal to clinical Parkinson's disease.

DeepSurv extends Cox regression with neural networks to:
- Model complex non-linear relationships between features and survival
- Capture interactions between biomarkers
- Provide flexible risk prediction

We compare DeepSurv performance against traditional Cox models from Task 5.3.

Key Features:
- Neural network-based Cox proportional hazards model
- Custom loss function (negative log partial likelihood)
- Feature importance via gradient-based attribution
- Risk score calibration
- Survival curve prediction for individual patients

Author: GIMAN Phase 5 Development
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class SurvivalDataset(Dataset):
    """PyTorch Dataset for survival data."""

    def __init__(self, X, T, E):
        """
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        T : np.ndarray
            Time to event (n_samples,)
        E : np.ndarray
            Event indicator (1=event, 0=censored) (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.T = torch.FloatTensor(T)
        self.E = torch.FloatTensor(E)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.E[idx]


class DeepSurv(nn.Module):
    """
    DeepSurv: Deep learning-based Cox proportional hazards model.

    Architecture:
    - Input layer: covariates
    - Hidden layers: fully connected with ReLU activation and dropout
    - Output layer: single neuron (log-hazard)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [32, 16],
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize DeepSurv model.

        Parameters
        ----------
        input_dim : int
            Number of input features
        hidden_dims : list
            List of hidden layer dimensions
        dropout : float
            Dropout probability
        activation : str
            Activation function ('relu', 'tanh', 'selu')
        """
        super(DeepSurv, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'selu':
                layers.append(nn.SELU())

            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        # Output layer (log-hazard)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Log-hazard predictions (batch_size, 1)
        """
        return self.network(x)


def cox_partial_likelihood_loss(log_h, T, E):
    """
    Negative log partial likelihood for Cox model.

    This is the loss function used to train DeepSurv.

    Parameters
    ----------
    log_h : torch.Tensor
        Log-hazard predictions (batch_size, 1)
    T : torch.Tensor
        Time to event (batch_size,)
    E : torch.Tensor
        Event indicator (batch_size,)

    Returns
    -------
    torch.Tensor
        Negative log partial likelihood (scalar)
    """
    # Sort by time (descending)
    sorted_idx = torch.argsort(T, descending=True)
    log_h = log_h[sorted_idx].squeeze()
    E = E[sorted_idx]

    # Compute log cumulative sum of exp(log_h)
    log_cumsum_h = torch.logcumsumexp(log_h, dim=0)

    # Compute log partial likelihood
    # Only consider uncensored events
    log_pl = torch.sum((log_h - log_cumsum_h) * E)

    # Return negative (for minimization)
    return -log_pl


def concordance_index(risk_scores, T, E):
    """
    Compute Harrell's concordance index (C-index).

    The C-index measures discrimination: the probability that,
    for a random pair of patients where one experienced an event,
    the model assigns higher risk to that patient.

    Parameters
    ----------
    risk_scores : np.ndarray
        Predicted risk scores
    T : np.ndarray
        Time to event
    E : np.ndarray
        Event indicator

    Returns
    -------
    float
        C-index (0.5 = random, 1.0 = perfect)
    """
    n = len(T)
    concordant = 0
    discordant = 0

    for i in range(n):
        if E[i] == 0:  # Censored
            continue

        for j in range(n):
            if i == j:
                continue

            # Consider pairs where i had event and j was at risk
            if T[j] >= T[i]:
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1

    if concordant + discordant == 0:
        return 0.5

    return concordant / (concordant + discordant)


class DeepSurvTrainer:
    """
    Trainer for DeepSurv model.
    """

    def __init__(
        self,
        model: DeepSurv,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : DeepSurv
            DeepSurv model
        learning_rate : float
            Learning rate
        weight_decay : float
            L2 regularization strength
        device : str
            Device ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        self.train_losses = []
        self.val_losses = []
        self.val_c_indices = []

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for X, T, E in dataloader:
            X = X.to(self.device)
            T = T.to(self.device)
            E = E.to(self.device)

            # Forward pass
            log_h = self.model(X)
            loss = cox_partial_likelihood_loss(log_h, T, E)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Evaluate model."""
        self.model.eval()
        all_risk_scores = []
        all_T = []
        all_E = []
        total_loss = 0.0

        with torch.no_grad():
            for X, T, E in dataloader:
                X = X.to(self.device)
                T_tensor = T.to(self.device)
                E_tensor = E.to(self.device)

                log_h = self.model(X)
                loss = cox_partial_likelihood_loss(log_h, T_tensor, E_tensor)
                total_loss += loss.item()

                # Store predictions
                risk_scores = torch.exp(log_h).cpu().numpy().flatten()
                all_risk_scores.extend(risk_scores)
                all_T.extend(T.numpy())
                all_E.extend(E.numpy())

        # Compute C-index
        c_index = concordance_index(
            np.array(all_risk_scores),
            np.array(all_T),
            np.array(all_E)
        )

        return total_loss / len(dataloader), c_index

    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 100,
        early_stopping_patience: int = 20
    ):
        """
        Train model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        n_epochs : int
            Maximum number of epochs
        early_stopping_patience : int
            Patience for early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0

        print("\nTraining DeepSurv model...")
        print(f"  Device: {self.device}")
        print(f"  Max epochs: {n_epochs}")
        print(f"  Early stopping patience: {early_stopping_patience}")

        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_c_index = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_c_indices.append(val_c_index)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"val_c_index={val_c_index:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\n  Training complete. Best val_loss: {best_val_loss:.4f}")


class DeepSurvAnalysis:
    """
    Complete DeepSurv analysis for prodromal phenoconversion prediction.
    """

    def __init__(
        self,
        survival_data_path: str,
        time_varying_data_path: str,
        output_dir: str,
        device: str = 'cpu'
    ):
        """
        Initialize DeepSurv analysis.

        Parameters
        ----------
        survival_data_path : str
            Path to baseline survival data
        time_varying_data_path : str
            Path to time-varying biomarker data
        output_dir : str
            Directory for saving outputs
        device : str
            Device for training ('cpu' or 'cuda')
        """
        self.survival_data_path = Path(survival_data_path)
        self.time_varying_data_path = Path(time_varying_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Load data
        print("Loading survival and time-varying data...")
        self.survival_df = pd.read_csv(self.survival_data_path)
        self.time_varying_df = pd.read_csv(self.time_varying_data_path)

        print(f"  Baseline data: {len(self.survival_df)} patients")
        print(f"  Events: {self.survival_df['phenoconverted'].sum()} phenoconversions")

        self.model = None
        self.trainer = None
        self.scaler = None
        self.results = {}

    def prepare_data(self):
        """Prepare data for DeepSurv training."""
        print("\nPreparing data for DeepSurv...")

        # Get slope data
        slope_df = self.time_varying_df.groupby('PATNO')[['UPDRS_III_slope']].last().reset_index()

        # Merge with baseline data
        df = self.survival_df.copy()
        df = df.merge(slope_df, on='PATNO', how='left')
        df['UPDRS_III_slope'] = df['UPDRS_III_slope'].fillna(0)

        # Select features
        feature_cols = [
            'baseline_updrs',
            'baseline_moca',
            'age_approx',
            'UPDRS_III_slope'
        ]

        # Handle missing values
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].median())

        # Add sex as binary feature
        df['sex_binary'] = (df['sex'] == 'M').astype(float)
        feature_cols.append('sex_binary')

        # Extract features, time, and events
        X = df[feature_cols].values
        T = df['time_to_event'].values
        E = df['phenoconverted'].astype(float).values

        print(f"  Features: {feature_cols}")
        print(f"  Shape: {X.shape}")
        print(f"  Events: {E.sum():.0f}/{len(E)} ({100*E.sum()/len(E):.1f}%)")

        # Train/validation split (80/20)
        # Stratify by event to ensure both sets have events
        X_train, X_val, T_train, T_val, E_train, E_val = train_test_split(
            X, T, E,
            test_size=0.2,
            random_state=42,
            stratify=E
        )

        print(f"\n  Train set: {len(X_train)} patients ({E_train.sum():.0f} events)")
        print(f"  Val set: {len(X_val)} patients ({E_val.sum():.0f} events)")

        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Create datasets
        train_dataset = SurvivalDataset(X_train_scaled, T_train, E_train)
        val_dataset = SurvivalDataset(X_val_scaled, T_val, E_val)

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False
        )

        # Store for later use
        self.X_val = X_val_scaled
        self.T_val = T_val
        self.E_val = E_val
        self.feature_names = feature_cols

        return X_train_scaled.shape[1]

    def build_and_train_model(self, input_dim):
        """Build and train DeepSurv model."""
        print("\nBuilding DeepSurv model...")

        # Initialize model
        self.model = DeepSurv(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            dropout=0.3,
            activation='relu'
        )

        print(f"  Architecture: {input_dim} -> 32 -> 16 -> 1")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters())}")

        # Initialize trainer
        self.trainer = DeepSurvTrainer(
            model=self.model,
            learning_rate=0.001,
            weight_decay=0.0001,
            device=self.device
        )

        # Train
        self.trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            n_epochs=200,
            early_stopping_patience=30
        )

        # Final evaluation
        final_val_loss, final_c_index = self.trainer.evaluate(self.val_loader)
        print(f"\nFinal validation performance:")
        print(f"  Loss: {final_val_loss:.4f}")
        print(f"  C-index: {final_c_index:.4f}")

        # Store results
        self.results['training'] = {
            'final_val_loss': float(final_val_loss),
            'final_c_index': float(final_c_index),
            'n_epochs': len(self.trainer.train_losses),
            'best_epoch': int(np.argmin(self.trainer.val_losses))
        }

    def compute_feature_importance(self):
        """Compute feature importance via gradient-based attribution."""
        print("\nComputing feature importance...")

        self.model.eval()

        # Use validation set
        X_tensor = torch.FloatTensor(self.X_val).to(self.device)
        X_tensor.requires_grad = True

        # Forward pass
        log_h = self.model(X_tensor)

        # Compute gradients
        log_h.sum().backward()

        # Feature importance = mean absolute gradient
        gradients = X_tensor.grad.detach().cpu().numpy()
        importance = np.abs(gradients).mean(axis=0)

        # Normalize
        importance = importance / importance.sum()

        # Store results
        self.results['feature_importance'] = {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance)
        }

        print("  Feature importance (normalized):")
        for name, imp in sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"    {name}: {imp:.4f}")

    def visualize_results(self):
        """Create comprehensive visualization of DeepSurv results."""
        print("\nCreating visualizations...")

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Training curves
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_training_curves(ax1)

        # 2. C-index progression
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_c_index_progression(ax2)

        # 3. Feature importance
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_feature_importance(ax3)

        # 4. Risk score distribution
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_risk_distribution(ax4)

        # 5. Survival curves by risk group
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_survival_by_risk(ax5)

        # 6. Calibration plot
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_calibration(ax6)

        # 7. Model comparison (DeepSurv vs Cox)
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_model_comparison(ax7)

        # 8. Risk score vs actual outcomes
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_risk_vs_outcomes(ax8)

        plt.suptitle(
            'DeepSurv Neural Survival Model Analysis\n'
            'Prodromal-to-Clinical PD Transition',
            fontsize=16, fontweight='bold', y=0.995
        )

        # Save figure
        output_path = self.output_dir / 'deepsurv_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    def _plot_training_curves(self, ax):
        """Plot training and validation loss curves."""
        epochs = range(1, len(self.trainer.train_losses) + 1)

        ax.plot(epochs, self.trainer.train_losses, label='Train Loss', linewidth=2)
        ax.plot(epochs, self.trainer.val_losses, label='Val Loss', linewidth=2)

        # Mark best epoch
        best_epoch = np.argmin(self.trainer.val_losses) + 1
        ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
                  label=f'Best (epoch {best_epoch})')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Negative Log Partial Likelihood')
        ax.set_title('Training Progress', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_c_index_progression(self, ax):
        """Plot C-index progression during training."""
        epochs = range(1, len(self.trainer.val_c_indices) + 1)

        ax.plot(epochs, self.trainer.val_c_indices, linewidth=2, color='green')

        # Mark best
        best_c = max(self.trainer.val_c_indices)
        best_epoch = self.trainer.val_c_indices.index(best_c) + 1
        ax.axhline(best_c, color='red', linestyle='--', alpha=0.5)
        ax.text(len(epochs)*0.7, best_c + 0.01, f'Best: {best_c:.4f}',
               fontsize=9, color='red')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('C-index')
        ax.set_title('Discrimination Performance', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance(self, ax):
        """Plot feature importance."""
        importance = self.results['feature_importance']

        names = list(importance.keys())
        values = list(importance.values())

        # Sort by importance
        sorted_idx = np.argsort(values)
        names = [names[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        ax.barh(range(len(names)), values, color=colors)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Importance (normalized)')
        ax.set_title('Feature Importance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_risk_distribution(self, ax):
        """Plot distribution of predicted risk scores."""
        # Get risk scores
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_val).to(self.device)
            log_h = self.model(X_tensor)
            risk_scores = torch.exp(log_h).cpu().numpy().flatten()

        # Plot histograms for events vs censored
        event_mask = self.E_val == 1

        ax.hist(risk_scores[~event_mask], bins=20, alpha=0.5, label='Censored', color='blue')
        ax.hist(risk_scores[event_mask], bins=10, alpha=0.7, label='Events', color='red')

        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Count')
        ax.set_title('Risk Score Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_survival_by_risk(self, ax):
        """Plot Kaplan-Meier curves by risk group."""
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            ax.text(
                0.5,
                0.5,
                "Kaplan-Meier plot unavailable\\n(`lifelines` not installed)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return

        # Get risk scores
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_val).to(self.device)
            log_h = self.model(X_tensor)
            risk_scores = torch.exp(log_h).cpu().numpy().flatten()

        # Create risk groups (tertiles)
        risk_tertiles = pd.qcut(risk_scores, q=3, labels=['Low', 'Medium', 'High'])

        # Fit KM for each group
        kmf = KaplanMeierFitter()

        for group in ['Low', 'Medium', 'High']:
            mask = risk_tertiles == group
            kmf.fit(
                self.T_val[mask],
                self.E_val[mask],
                label=f'{group} Risk (n={mask.sum()})'
            )
            kmf.plot_survival_function(ax=ax, ci_show=True)

        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('KM Curves by Risk Group', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_calibration(self, ax):
        """Plot calibration curve."""
        # Simple calibration: predicted risk vs observed event rate in bins
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_val).to(self.device)
            log_h = self.model(X_tensor)
            risk_scores = torch.exp(log_h).cpu().numpy().flatten()

        # Create bins
        n_bins = 5
        bin_edges = np.percentile(risk_scores, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        observed_rates = []

        for i in range(n_bins):
            mask = (risk_scores >= bin_edges[i]) & (risk_scores < bin_edges[i+1])
            if i == n_bins - 1:  # Include upper edge in last bin
                mask = (risk_scores >= bin_edges[i]) & (risk_scores <= bin_edges[i+1])

            if mask.sum() > 0:
                bin_centers.append(risk_scores[mask].mean())
                observed_rates.append(self.E_val[mask].mean())

        # Plot
        ax.scatter(bin_centers, observed_rates, s=100, alpha=0.7, color='blue', label='Observed')

        # Perfect calibration line
        max_val = max(max(bin_centers), max(observed_rates))
        ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect Calibration')

        ax.set_xlabel('Predicted Risk')
        ax.set_ylabel('Observed Event Rate')
        ax.set_title('Calibration Plot', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_comparison(self, ax):
        """Compare DeepSurv vs Cox models."""
        # Load Cox results
        cox_results_path = self.output_dir / 'cox_model_results.json'
        if cox_results_path.exists():
            with open(cox_results_path) as f:
                cox_results = json.load(f)

            models = ['Cox\nBaseline', 'Cox\n+Slope', 'DeepSurv']
            c_indices = [
                cox_results['baseline_model']['concordance_index'],
                cox_results['time_varying_model']['concordance_index'],
                self.results['training']['final_c_index']
            ]

            colors = ['steelblue', 'darkorange', 'forestgreen']
            bars = ax.bar(range(len(models)), c_indices, color=colors, alpha=0.7)

            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars, c_indices)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models)
            ax.set_ylabel('C-index')
            ax.set_title('Model Comparison', fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Cox results\nnot available',
                   ha='center', va='center', transform=ax.transAxes)

    def _plot_risk_vs_outcomes(self, ax):
        """Plot risk scores vs actual outcomes."""
        # Get risk scores
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_val).to(self.device)
            log_h = self.model(X_tensor)
            risk_scores = torch.exp(log_h).cpu().numpy().flatten()

        # Scatter plot
        event_mask = self.E_val == 1

        ax.scatter(self.T_val[~event_mask], risk_scores[~event_mask],
                  alpha=0.5, s=50, label='Censored', color='blue', marker='o')
        ax.scatter(self.T_val[event_mask], risk_scores[event_mask],
                  alpha=0.7, s=100, label='Events', color='red', marker='X')

        ax.set_xlabel('Time to Event/Censoring (months)')
        ax.set_ylabel('Predicted Risk Score')
        ax.set_title('Risk Predictions vs Outcomes', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def save_outputs(self):
        """Save analysis results."""
        print("\nSaving results...")

        # Add metadata
        self.results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'n_patients': int(len(self.survival_df)),
            'n_events': int(self.survival_df['phenoconverted'].sum()),
            'device': self.device,
            'input_dim': len(self.feature_names),
            'features': self.feature_names
        }

        # Save results JSON
        results_path = self.output_dir / 'deepsurv_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  Saved: {results_path}")

        # Save model
        model_path = self.output_dir / 'deepsurv_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'feature_names': self.feature_names
        }, model_path)
        print(f"  Saved: {model_path}")

    def run_complete_analysis(self):
        """Execute complete DeepSurv analysis pipeline."""
        print("\n" + "="*80)
        print("DEEPSURV NEURAL SURVIVAL ANALYSIS")
        print("="*80)

        # 1. Prepare data
        input_dim = self.prepare_data()

        # 2. Build and train model
        self.build_and_train_model(input_dim)

        # 3. Compute feature importance
        self.compute_feature_importance()

        # 4. Visualize
        self.visualize_results()

        # 5. Save outputs
        self.save_outputs()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nOutputs saved to: {self.output_dir}")


def main():
    """Main execution function."""

    # Define paths relative to project root
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "prodromal_cohort"

    survival_data_path = data_dir / "prodromal_survival_data.csv"
    time_varying_data_path = data_dir / "time_varying_biomarkers.csv"
    output_dir = data_dir

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Run analysis
    analyzer = DeepSurvAnalysis(
        survival_data_path=str(survival_data_path),
        time_varying_data_path=str(time_varying_data_path),
        output_dir=str(output_dir),
        device=device
    )

    analyzer.run_complete_analysis()

    print("\n[SUCCESS] Task 5.4 Complete: DeepSurv Neural Survival Model")


if __name__ == "__main__":
    main()
