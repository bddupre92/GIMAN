"""
Phase 4, Task 4.3: Trajectory Clustering with VaDER

This script implements trajectory clustering to discover distinct PD progression subtypes using:
1. VaDER (Variational Deep Embedding with Recurrence) - deep learning approach
2. K-Means clustering on trajectory features - baseline approach
3. Hierarchical clustering - exploratory approach

Methodology:
- Encode aligned disease-time trajectories into latent embeddings
- Cluster embeddings to identify 3-5 distinct progression subtypes
- Characterize subtypes by motor/cognitive progression patterns

Expected Output:
- Subtype assignments for each patient
- Cluster centroids and characteristics
- Visualization of subtype separation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory sequences."""

    def __init__(self, sequences: np.ndarray, patient_ids: np.ndarray):
        """
        Args:
            sequences: (N, max_seq_len, n_features) array
            patient_ids: (N,) array of patient IDs
        """
        self.sequences = torch.FloatTensor(sequences)
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.patient_ids[idx]


class VaDEREncoder(nn.Module):
    """
    Variational Deep Embedding with Recurrence (VaDER) encoder.

    Architecture:
    - Bidirectional LSTM to encode temporal sequences
    - Variational layer for latent representation
    - Produces fixed-dimensional embedding for each trajectory
    """

    def __init__(
        self,
        input_dim: int = 2,  # UPDRS-III and MoCA
        hidden_dim: int = 32,
        latent_dim: int = 16,
        num_layers: int = 2
    ):
        super(VaDEREncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Variational layers (mean and log-variance)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim) tensor
            lengths: actual sequence lengths (for padding)

        Returns:
            mu: (batch, latent_dim) mean of latent distribution
            logvar: (batch, latent_dim) log-variance of latent distribution
            z: (batch, latent_dim) sampled latent vector
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state (concatenate forward and backward)
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        h_forward = h_n[-2, :, :]  # Last layer forward
        h_backward = h_n[-1, :, :]  # Last layer backward
        h_final = torch.cat([h_forward, h_backward], dim=1)

        # Variational parameters
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return mu, logvar, z


class VaDERDecoder(nn.Module):
    """VaDER decoder for trajectory reconstruction."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        output_dim: int = 2,
        seq_len: int = 3
    ):
        super(VaDERDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len

        # Expand latent to hidden
        self.fc_expand = nn.Linear(latent_dim, hidden_dim)

        # LSTM decoder
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim) latent vectors

        Returns:
            x_recon: (batch, seq_len, output_dim) reconstructed sequences
        """
        batch_size = z.size(0)

        # Expand latent
        h = torch.relu(self.fc_expand(z))

        # Repeat for sequence length
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)

        # LSTM decoding
        lstm_out, _ = self.lstm(h)

        # Output
        x_recon = self.fc_out(lstm_out)

        return x_recon


class VaDER(nn.Module):
    """Complete VaDER model (encoder + decoder)."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        seq_len: int = 3
    ):
        super(VaDER, self).__init__()

        self.encoder = VaDEREncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VaDERDecoder(latent_dim, hidden_dim, input_dim, seq_len)

    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def encode(self, x):
        """Get latent embeddings (no gradient)."""
        with torch.no_grad():
            mu, logvar, z = self.encoder(x)
        return z.cpu().numpy()


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE loss = reconstruction loss + KL divergence.

    Args:
        x_recon: reconstructed sequences
        x: original sequences
        mu: latent mean
        logvar: latent log-variance
        beta: KL weighting factor
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss


class TrajectoryClusteringPipeline:
    """Complete trajectory clustering pipeline."""

    def __init__(
        self,
        aligned_observations_path: str,
        trajectories_path: str,
        output_dir: str = "data/longitudinal_cohort",
        n_clusters_range: Tuple[int, int] = (3, 6),
        latent_dim: int = 16,
        n_epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize trajectory clustering pipeline.

        Args:
            aligned_observations_path: Path to aligned_observations.csv
            trajectories_path: Path to patient_trajectories_aligned.csv
            output_dir: Directory for outputs
            n_clusters_range: Range of cluster numbers to evaluate
            latent_dim: VaDER latent dimension
            n_epochs: Training epochs for VaDER
            batch_size: Batch size for VaDER training
        """
        self.aligned_obs_path = Path(aligned_observations_path)
        self.trajectories_path = Path(trajectories_path)
        self.output_dir = Path(output_dir)

        self.n_clusters_range = n_clusters_range
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.aligned_df = None
        self.trajectories_df = None
        self.sequences = None
        self.patient_ids = None

        self.vader_model = None
        self.embeddings = None
        self.cluster_labels = None
        self.optimal_n_clusters = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("[INIT] Initialized Trajectory Clustering Pipeline")
        print(f"   Aligned observations: {self.aligned_obs_path}")
        print(f"   Trajectories: {self.trajectories_path}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Cluster range: {n_clusters_range}")
        print(f"   Latent dimension: {latent_dim}")
        print(f"   Device: {self.device}")

    def load_data(self):
        """Load aligned trajectory data."""
        print("\n[LOAD] Loading aligned trajectory data...")

        self.aligned_df = pd.read_csv(self.aligned_obs_path)
        self.trajectories_df = pd.read_csv(self.trajectories_path)

        print(f"   Loaded {len(self.aligned_df)} observations")
        print(f"   Loaded {len(self.trajectories_df)} patient trajectories")

        return self.aligned_df, self.trajectories_df

    def prepare_sequences(self):
        """
        Prepare trajectory sequences for VaDER.

        Returns:
            sequences: (N, max_seq_len, 2) array of [UPDRS-III, MoCA] sequences
            patient_ids: (N,) array of patient IDs
        """
        print("\n[PREP] Preparing trajectory sequences...")

        # Get unique patients
        patient_ids = self.aligned_df['PATNO'].unique()

        # Determine max sequence length
        seq_lengths = self.aligned_df.groupby('PATNO').size()
        max_seq_len = int(seq_lengths.max())

        print(f"   Unique patients: {len(patient_ids)}")
        print(f"   Max sequence length: {max_seq_len}")

        # Build sequences
        sequences_list = []
        valid_patient_ids = []

        for patno in patient_ids:
            patient_data = self.aligned_df[
                self.aligned_df['PATNO'] == patno
            ].sort_values('disease_time')

            # Extract UPDRS-III and MoCA
            updrs = patient_data['UPDRS_III'].values
            moca = patient_data['MOCA'].values

            # Skip if insufficient data
            if len(updrs) < 2 or np.isnan(updrs).all() or np.isnan(moca).all():
                continue

            # Handle missing values (forward fill)
            updrs = pd.Series(updrs).fillna(method='ffill').fillna(method='bfill').values
            moca = pd.Series(moca).fillna(method='ffill').fillna(method='bfill').values

            # Create sequence (pad if necessary)
            seq = np.zeros((max_seq_len, 2))
            actual_len = min(len(updrs), max_seq_len)

            seq[:actual_len, 0] = updrs[:actual_len]
            seq[:actual_len, 1] = moca[:actual_len]

            # Pad remaining with last observation (if needed)
            if actual_len < max_seq_len:
                seq[actual_len:, 0] = updrs[actual_len - 1]
                seq[actual_len:, 1] = moca[actual_len - 1]

            sequences_list.append(seq)
            valid_patient_ids.append(patno)

        self.sequences = np.array(sequences_list)
        self.patient_ids = np.array(valid_patient_ids)

        print(f"   Created {len(self.sequences)} sequences")
        print(f"   Sequence shape: {self.sequences.shape}")

        # Normalize sequences
        self.scaler = StandardScaler()

        # Reshape for scaling: (N * seq_len, 2)
        N, seq_len, n_features = self.sequences.shape
        sequences_reshaped = self.sequences.reshape(-1, n_features)
        sequences_scaled = self.scaler.fit_transform(sequences_reshaped)
        self.sequences = sequences_scaled.reshape(N, seq_len, n_features)

        print(f"   Normalized sequences (mean=0, std=1)")

        return self.sequences, self.patient_ids

    def train_vader(self):
        """Train VaDER model for trajectory embedding."""
        print("\n[VADER] Training VaDER model...")

        # Create dataset and dataloader
        dataset = TrajectoryDataset(self.sequences, self.patient_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        seq_len = self.sequences.shape[1]
        self.vader_model = VaDER(
            input_dim=2,
            hidden_dim=32,
            latent_dim=self.latent_dim,
            seq_len=seq_len
        ).to(self.device)

        # Optimizer
        optimizer = optim.Adam(self.vader_model.parameters(), lr=1e-3)

        # Training loop
        self.vader_model.train()
        losses = []

        for epoch in range(self.n_epochs):
            epoch_loss = 0

            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)

                # Forward pass
                x_recon, mu, logvar, z = self.vader_model(batch_x)

                # Loss
                loss = vae_loss(x_recon, batch_x, mu, logvar, beta=0.1)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{self.n_epochs}: Loss = {avg_loss:.2f}")

        print(f"   Training complete. Final loss: {losses[-1]:.2f}")

        # Extract embeddings
        self.vader_model.eval()
        all_sequences = torch.FloatTensor(self.sequences).to(self.device)
        self.embeddings = self.vader_model.encode(all_sequences)

        print(f"   Extracted embeddings: {self.embeddings.shape}")

        return self.embeddings

    def determine_optimal_clusters(self) -> int:
        """
        Determine optimal number of clusters using multiple metrics.

        Returns:
            Optimal number of clusters
        """
        print("\n[CLUSTER] Determining optimal number of clusters...")

        metrics_results = {
            'n_clusters': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }

        for n in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=20)
            labels = kmeans.fit_predict(self.embeddings)

            sil_score = silhouette_score(self.embeddings, labels)
            db_score = davies_bouldin_score(self.embeddings, labels)
            ch_score = calinski_harabasz_score(self.embeddings, labels)

            metrics_results['n_clusters'].append(n)
            metrics_results['silhouette'].append(sil_score)
            metrics_results['davies_bouldin'].append(db_score)
            metrics_results['calinski_harabasz'].append(ch_score)

            print(f"   n={n}: Silhouette={sil_score:.3f}, DB={db_score:.3f}, CH={ch_score:.1f}")

        # Choose optimal n (maximize silhouette)
        optimal_idx = np.argmax(metrics_results['silhouette'])
        self.optimal_n_clusters = metrics_results['n_clusters'][optimal_idx]

        print(f"\n   Optimal clusters: {self.optimal_n_clusters} (max silhouette)")

        return self.optimal_n_clusters, metrics_results

    def cluster_trajectories(self):
        """Perform final clustering with optimal number of clusters."""
        print(f"\n[CLUSTER] Clustering with k={self.optimal_n_clusters}...")

        # K-Means clustering
        kmeans = KMeans(n_clusters=self.optimal_n_clusters, random_state=42, n_init=20)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)

        # Add to trajectories dataframe
        cluster_df = pd.DataFrame({
            'PATNO': self.patient_ids,
            'cluster': self.cluster_labels
        })

        self.trajectories_df = self.trajectories_df.merge(cluster_df, on='PATNO', how='left')

        # Print cluster sizes
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        print(f"\n   Cluster sizes:")
        for cluster_id, count in cluster_counts.items():
            pct = 100 * count / len(self.cluster_labels)
            print(f"      Cluster {cluster_id}: {count} patients ({pct:.1f}%)")

        return self.cluster_labels

    def generate_visualizations(self, metrics_results):
        """Generate trajectory clustering visualizations."""
        print("\n[VIZ] Generating clustering visualizations...")

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

        fig.suptitle('Phase 4, Task 4.3: Trajectory Clustering Analysis (VaDER)',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Cluster selection metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(metrics_results['n_clusters'], metrics_results['silhouette'],
                'o-', color='steelblue', linewidth=2, markersize=8)
        ax1.axvline(self.optimal_n_clusters, color='red', linestyle='--',
                   label=f'Optimal: k={self.optimal_n_clusters}')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score (higher is better)')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Davies-Bouldin Index
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(metrics_results['n_clusters'], metrics_results['davies_bouldin'],
                'o-', color='coral', linewidth=2, markersize=8)
        ax2.axvline(self.optimal_n_clusters, color='red', linestyle='--')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_title('Davies-Bouldin Index (lower is better)')
        ax2.grid(alpha=0.3)

        # 3. PCA visualization of embeddings (2D)
        ax3 = fig.add_subplot(gs[0, 2])
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings)

        scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             c=self.cluster_labels, cmap='tab10', alpha=0.6, s=50)
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax3.set_title('VaDER Embeddings (PCA projection)')
        plt.colorbar(scatter, ax=ax3, label='Cluster')
        ax3.grid(alpha=0.3)

        # 4. Cluster sizes
        ax4 = fig.add_subplot(gs[0, 3])
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        colors = plt.cm.tab10(np.arange(len(cluster_counts)))
        ax4.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Number of Patients')
        ax4.set_title('Cluster Size Distribution')
        ax4.grid(axis='y', alpha=0.3)

        # 5-8. Motor progression by cluster
        for i in range(self.optimal_n_clusters):
            row = 1 + i // 4
            col = i % 4
            ax = fig.add_subplot(gs[row, col])

            # Get patients in this cluster
            cluster_patients = self.patient_ids[self.cluster_labels == i]

            # Plot trajectories
            for patno in cluster_patients[:30]:  # Max 30 per cluster for visibility
                patient_data = self.aligned_df[
                    self.aligned_df['PATNO'] == patno
                ].dropna(subset=['UPDRS_III']).sort_values('disease_time')

                if len(patient_data) > 1:
                    ax.plot(patient_data['disease_time'] / 12,
                           patient_data['UPDRS_III'],
                           alpha=0.3, linewidth=1, color=colors[i])

            # Cluster mean trajectory
            cluster_data = self.aligned_df[
                self.aligned_df['PATNO'].isin(cluster_patients)
            ].dropna(subset=['UPDRS_III', 'disease_time'])

            if len(cluster_data) > 0:
                # Bin by disease time
                bins = np.linspace(0, cluster_data['disease_time'].max() / 12, 20)
                cluster_data['disease_time_binned'] = pd.cut(
                    cluster_data['disease_time'] / 12, bins, include_lowest=True
                )

                mean_traj = cluster_data.groupby('disease_time_binned')['UPDRS_III'].mean()
                bin_centers = [(interval.left + interval.right) / 2 for interval in mean_traj.index]

                ax.plot(bin_centers, mean_traj.values, 'k-', linewidth=3,
                       label=f'Mean (n={len(cluster_patients)})')

            ax.set_xlabel('Disease Time (years)')
            ax.set_ylabel('UPDRS-III Score')
            ax.set_title(f'Cluster {i} Motor Trajectories')
            ax.legend()
            ax.grid(alpha=0.3)

        # Save figure
        viz_path = self.output_dir / 'trajectory_clustering_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_path}")

        plt.close()

    def save_outputs(self):
        """Save clustering results."""
        print("\n[SAVE] Saving clustering outputs...")

        # Save trajectories with cluster assignments
        traj_path = self.output_dir / 'patient_trajectories_clustered.csv'
        self.trajectories_df.to_csv(traj_path, index=False)
        print(f"   Saved clustered trajectories: {traj_path}")

        # Save embeddings
        embeddings_df = pd.DataFrame(
            self.embeddings,
            columns=[f'emb_{i}' for i in range(self.embeddings.shape[1])]
        )
        embeddings_df['PATNO'] = self.patient_ids
        embeddings_df['cluster'] = self.cluster_labels

        emb_path = self.output_dir / 'vader_embeddings.csv'
        embeddings_df.to_csv(emb_path, index=False)
        print(f"   Saved VaDER embeddings: {emb_path}")

        # Save cluster summary
        cluster_summary = []
        for cluster_id in range(self.optimal_n_clusters):
            cluster_patients = self.patient_ids[self.cluster_labels == cluster_id]
            cluster_traj = self.trajectories_df[
                self.trajectories_df['PATNO'].isin(cluster_patients)
            ]

            summary = {
                'cluster_id': int(cluster_id),
                'n_patients': int(len(cluster_patients)),
                'percentage': float(100 * len(cluster_patients) / len(self.patient_ids)),
                'mean_updrs_slope': float(cluster_traj['UPDRS_III_slope'].mean()),
                'std_updrs_slope': float(cluster_traj['UPDRS_III_slope'].std()),
                'mean_moca_slope': float(cluster_traj['MOCA_slope'].mean()),
                'std_moca_slope': float(cluster_traj['MOCA_slope'].std()),
                'mean_updrs_baseline': float(cluster_traj['UPDRS_III_baseline'].mean()),
                'mean_moca_baseline': float(cluster_traj['MOCA_baseline'].mean())
            }

            cluster_summary.append(summary)

        summary_report = {
            'cluster_summary': cluster_summary,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'optimal_n_clusters': int(self.optimal_n_clusters),
                'latent_dim': int(self.latent_dim),
                'n_epochs': int(self.n_epochs),
                'total_patients': int(len(self.patient_ids))
            }
        }

        report_path = self.output_dir / 'clustering_report.json'
        with open(report_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        print(f"   Saved clustering report: {report_path}")

    def run_full_pipeline(self):
        """Execute complete trajectory clustering pipeline."""
        print("="*80)
        print("PHASE 4, TASK 4.3: TRAJECTORY CLUSTERING")
        print("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Prepare sequences
        self.prepare_sequences()

        # Step 3: Train VaDER
        self.train_vader()

        # Step 4: Determine optimal clusters
        optimal_n, metrics_results = self.determine_optimal_clusters()

        # Step 5: Cluster trajectories
        self.cluster_trajectories()

        # Step 6: Generate visualizations
        self.generate_visualizations(metrics_results)

        # Step 7: Save outputs
        self.save_outputs()

        print("\n" + "="*80)
        print("TASK 4.3 COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"   Total patients: {len(self.patient_ids)}")
        print(f"   Optimal clusters: {self.optimal_n_clusters}")
        print(f"   VaDER latent dimension: {self.latent_dim}")
        print(f"   Embedding shape: {self.embeddings.shape}")
        print(f"\nOutputs saved to: {self.output_dir}")

        return self.cluster_labels, self.embeddings, self.trajectories_df


def main():
    """Main execution function."""

    # Define paths relative to project root
    project_root = Path(__file__).resolve().parents[3]
    aligned_obs_path = project_root / "data" / "longitudinal_cohort" / "aligned_observations.csv"
    trajectories_path = project_root / "data" / "longitudinal_cohort" / "patient_trajectories_aligned.csv"
    output_dir = project_root / "data" / "longitudinal_cohort"
    N_CLUSTERS_RANGE = (3, 6)
    LATENT_DIM = 16
    N_EPOCHS = 100
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize and run pipeline
    pipeline = TrajectoryClusteringPipeline(
        aligned_observations_path=aligned_obs_path,
        trajectories_path=trajectories_path,
        output_dir=output_dir,
        n_clusters_range=N_CLUSTERS_RANGE,
        latent_dim=LATENT_DIM,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE
    )

    labels, embeddings, trajectories_df = pipeline.run_full_pipeline()

    return labels, embeddings, trajectories_df


if __name__ == "__main__":
    labels, embeddings, trajectories_df = main()
