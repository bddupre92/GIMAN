#!/usr/bin/env python3
"""
Enhanced Feature Dataset Creator for GIMAN v1.1.0
================================================

Creates enhanced 12-feature dataset by extending the current 7-feature production model
with additional biomarkers while maintaining full compatibility with existing architecture.

Architecture Integration:
- Follows existing preprocessing patterns from src/giman_pipeline/data_processing/
- Uses established biomarker features from integrate_biomarker_data.py
- Maintains compatibility with GIMANDataLoader and PyTorch Geometric format
- Preserves production model graph structure (557 nodes, same connectivity)

Enhanced Features Strategy:
- Current 7: LRRK2, GBA, APOE_RISK, PTAU, TTAU, UPSIT_TOTAL, ALPHA_SYN (biomarker features)
- Enhanced +5: AGE_COMPUTED, NHY, SEX, NP3TOT, HAS_DATSCAN (clinical/demographic features)

Longitudinal Data:
- Dataset contains multiple visits per patient (longitudinal follow-up)
- Patients may have varying clinical scores (NP3TOT, NHY) across visits
- All rows preserved to maintain temporal clinical progression information

Author: GIMAN Enhancement Team
Date: September 24, 2025
Version: 1.0.0
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EnhancedFeatureMapper:
    """Maps and validates enhanced features for GIMAN v1.1.0."""
    
    def __init__(self):
        """Initialize feature mapping definitions."""
        # Current production features (exactly as used in production model)
        self.current_features = [
            'LRRK2', 'GBA', 'APOE_RISK', 'PTAU', 'TTAU', 'UPSIT_TOTAL', 'ALPHA_SYN'
        ]
        
        # Enhancement features to add
        self.enhancement_features = [
            'AGE_COMPUTED', 'NHY', 'SEX', 'NP3TOT', 'HAS_DATSCAN'
        ]
        
        # Complete enhanced feature set
        self.all_features = self.current_features + self.enhancement_features
        
        # Feature type mapping for proper preprocessing
        self.feature_types = {
            # Current production features (biomarkers)
            'LRRK2': 'continuous',      # LRRK2 mutation status (continuous encoded)
            'GBA': 'continuous',        # GBA mutation status (continuous encoded)
            'APOE_RISK': 'continuous',  # APOE risk score (continuous)
            'PTAU': 'continuous',       # CSF phosphorylated tau
            'TTAU': 'continuous',       # CSF total tau
            'UPSIT_TOTAL': 'continuous', # UPSIT smell test total score
            'ALPHA_SYN': 'continuous',  # CSF alpha-synuclein levels
            # Enhancement features 
            'AGE_COMPUTED': 'continuous', # Patient age
            'NHY': 'ordinal',           # Hoehn & Yahr stage 0-5
            'SEX': 'binary',            # Gender (0=female, 1=male)
            'NP3TOT': 'continuous',     # UPDRS Part III motor total score
            'HAS_DATSCAN': 'binary'     # DaTScan availability flag
        }
        
    def validate_features(self, dataset: pd.DataFrame) -> Dict[str, Dict]:
        """Validate feature availability and coverage in dataset."""
        validation_results = {}
        
        for feature in self.all_features:
            if feature in dataset.columns:
                # Calculate coverage statistics
                total_count = len(dataset)
                non_null_count = dataset[feature].notna().sum()
                coverage = (non_null_count / total_count) * 100
                
                # Get value distribution
                if self.feature_types[feature] in ['binary', 'ordinal']:
                    value_counts = dataset[feature].value_counts().to_dict()
                else:
                    value_counts = {
                        'min': float(dataset[feature].min()),
                        'max': float(dataset[feature].max()),
                        'mean': float(dataset[feature].mean()),
                        'std': float(dataset[feature].std())
                    }
                
                validation_results[feature] = {
                    'available': True,
                    'coverage': coverage,
                    'non_null_count': non_null_count,
                    'total_count': total_count,
                    'feature_type': self.feature_types[feature],
                    'value_distribution': value_counts
                }
            else:
                validation_results[feature] = {
                    'available': False,
                    'coverage': 0.0,
                    'feature_type': self.feature_types[feature],
                    'status': 'missing_from_dataset'
                }
                
        return validation_results


class EnhancedDatasetCreator:
    """Creates enhanced 12-feature dataset for GIMAN v1.1.0."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "data/enhanced"):
        """
        Initialize enhanced dataset creator.
        
        Args:
            data_dir: Root data directory
            output_dir: Output directory for enhanced datasets
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "01_processed"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature mapper
        self.feature_mapper = EnhancedFeatureMapper()
        
        # Data containers
        self.original_dataset = None
        self.enhanced_dataset = None
        self.production_graph_data = None
        
        print(f"🚀 Enhanced Dataset Creator v1.1.0 Initialized")
        print(f"📂 Processed Directory: {self.processed_dir}")
        print(f"📂 Output Directory: {self.output_dir}")
        print(f"📊 Target Features: {len(self.feature_mapper.all_features)}")
        
    def load_production_graph_data(self) -> Data:
        """Load current production model's graph data for structure reference."""
        print("\n📥 Loading production graph data structure...")
        
        graph_path = Path("models/registry/giman_binary_classifier_v1.0.0/graph_data.pth")
        
        if not graph_path.exists():
            raise FileNotFoundError(f"Production graph data not found at {graph_path}")
            
        # Load with proper torch_geometric imports
        torch.serialization.add_safe_globals([Data])
        self.production_graph_data = torch.load(graph_path, weights_only=False)
        
        print(f"✅ Production graph loaded:")
        print(f"   📊 Nodes: {self.production_graph_data.num_nodes}")
        print(f"   📊 Edges: {self.production_graph_data.num_edges}")  
        print(f"   📊 Features: {self.production_graph_data.x.shape[1]}")
        print(f"   📊 Labels: {self.production_graph_data.y.shape}")
        
        return self.production_graph_data
        
    def load_enhanced_source_data(self) -> pd.DataFrame:
        """Load enhanced source dataset with biomarker features."""
        print("\n📥 Loading enhanced source dataset...")
        
        # Look for biomarker-enhanced datasets first
        enhanced_files = [
            "giman_biomarker_imputed_557_patients_v1.csv",
            "giman_enhanced_with_alpha_syn.csv",
            "giman_dataset_enhanced.csv",
            "giman_dataset_enriched.csv"
        ]
        
        source_dataset = None
        for filename in enhanced_files:
            filepath = self.processed_dir / filename
            if filepath.exists():
                print(f"✅ Found enhanced dataset: {filename}")
                source_dataset = pd.read_csv(filepath)
                print(f"📊 Shape: {source_dataset.shape}")
                print(f"📊 Columns: {list(source_dataset.columns)}")
                break
                
        if source_dataset is None:
            raise FileNotFoundError(
                f"No enhanced source dataset found in {self.processed_dir}. "
                f"Looking for: {enhanced_files}"
            )
            
        self.original_dataset = source_dataset
        return source_dataset
        
    def map_current_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Extract current 7 biomarker features that production model uses."""
        print("\n🔧 Extracting current 7 biomarker features from production model...")
        
        mapped_features = pd.DataFrame()
        mapping_results = {}
        
        # Copy PATNO if available
        if 'PATNO' in dataset.columns:
            mapped_features['PATNO'] = dataset['PATNO']
            
        # Copy labels/cohort info  
        label_cols = ['COHORT_DEFINITION', 'labels', 'y']
        for col in label_cols:
            if col in dataset.columns:
                mapped_features[col] = dataset[col]
                break
        
        # Extract each current biomarker feature (these should already exist in the dataset)
        for feature in self.feature_mapper.current_features:
            if feature in dataset.columns:
                mapped_features[feature] = dataset[feature]
                coverage = (dataset[feature].notna().sum() / len(dataset)) * 100
                mapping_results[feature] = {
                    'source_column': feature,
                    'coverage': coverage
                }
                print(f"   ✅ {feature}: {coverage:.1f}% coverage")
            else:
                print(f"   ❌ {feature}: Not found in dataset")
                mapping_results[feature] = {'source_column': None, 'coverage': 0.0}
                
        return mapped_features, mapping_results
        
    def extract_enhancement_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Extract the 5 enhancement features from source dataset."""
        print("\n🔧 Extracting enhancement features...")
        
        enhancement_data = pd.DataFrame()
        
        # Copy PATNO for merging
        if 'PATNO' in dataset.columns:
            enhancement_data['PATNO'] = dataset['PATNO']
            
        extraction_results = {}
        
        # Extract each enhancement feature
        for feature in self.feature_mapper.enhancement_features:
            if feature in dataset.columns:
                enhancement_data[feature] = dataset[feature]
                coverage = (dataset[feature].notna().sum() / len(dataset)) * 100
                extraction_results[feature] = {
                    'available': True,
                    'coverage': coverage,
                    'unique_values': len(dataset[feature].dropna().unique())
                }
                print(f"   ✅ {feature}: {coverage:.1f}% coverage, {extraction_results[feature]['unique_values']} unique values")
            else:
                print(f"   ❌ {feature}: Not found in source dataset")
                extraction_results[feature] = {'available': False, 'coverage': 0.0}
                
        return enhancement_data, extraction_results
        
    def create_enhanced_feature_matrix(self) -> Tuple[pd.DataFrame, Dict]:
        """Create the complete 12-feature enhanced dataset."""
        print("\n🔄 Creating enhanced 12-feature dataset...")
        
        # Load source data
        source_dataset = self.load_enhanced_source_data()
        
        # Map current 7 features
        current_features_df, current_mapping = self.map_current_features(source_dataset)
        
        # Extract 5 enhancement features  
        enhancement_features_df, enhancement_mapping = self.extract_enhancement_features(source_dataset)
        
        # Merge current and enhancement features
        if 'PATNO' in current_features_df.columns and 'PATNO' in enhancement_features_df.columns:
            enhanced_df = pd.merge(current_features_df, enhancement_features_df, on='PATNO', how='inner')
        else:
            # If no PATNO, assume same order (risky but fallback)
            enhanced_df = pd.concat([current_features_df, enhancement_features_df], axis=1)
            
        print(f"📊 Enhanced dataset shape: {enhanced_df.shape}")
        
        # Preserve all rows - this is longitudinal data with multiple visits per patient
        if 'PATNO' in enhanced_df.columns:
            unique_patients = enhanced_df['PATNO'].nunique()
            total_visits = len(enhanced_df)
            avg_visits = total_visits / unique_patients
            print(f"📊 Longitudinal data: {unique_patients} patients, {total_visits} visits (avg: {avg_visits:.1f} visits/patient)")
        else:
            print("⚠️ No PATNO column found")
        
        # Validate final feature set
        validation_results = self.feature_mapper.validate_features(enhanced_df)
        
        # Create comprehensive metadata
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'source_dataset_shape': source_dataset.shape,
            'enhanced_dataset_shape': enhanced_df.shape,
            'current_features': self.feature_mapper.current_features,
            'enhancement_features': self.feature_mapper.enhancement_features,
            'all_features': self.feature_mapper.all_features,
            'current_feature_mapping': current_mapping,
            'enhancement_feature_extraction': enhancement_mapping,
            'feature_validation': validation_results,
            'feature_types': self.feature_mapper.feature_types
        }
        
        self.enhanced_dataset = enhanced_df
        return enhanced_df, metadata
        
    def impute_missing_values(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Impute missing values using feature-type-appropriate strategies."""
        print("\n🔧 Imputing missing values with feature-type-aware strategies...")
        
        imputed_df = dataset.copy()
        imputation_stats = {}
        
        # Separate features by type for appropriate imputation
        continuous_features = [f for f in self.feature_mapper.all_features 
                             if f in dataset.columns and 
                             self.feature_mapper.feature_types[f] == 'continuous']
        
        categorical_features = [f for f in self.feature_mapper.all_features
                              if f in dataset.columns and 
                              self.feature_mapper.feature_types[f] in ['binary', 'ordinal']]
        
        # Impute continuous features with KNN (better for biomarkers)
        if continuous_features:
            print(f"   🔄 Imputing {len(continuous_features)} continuous features with KNN...")
            continuous_data = imputed_df[continuous_features]
            missing_before = continuous_data.isnull().sum().sum()
            
            if missing_before > 0:
                knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
                imputed_continuous = knn_imputer.fit_transform(continuous_data)
                imputed_df[continuous_features] = imputed_continuous
                
                imputation_stats['continuous'] = {
                    'features': continuous_features,
                    'missing_before': int(missing_before),
                    'missing_after': int(imputed_df[continuous_features].isnull().sum().sum()),
                    'method': 'KNN_k5_distance_weighted'
                }
            else:
                imputation_stats['continuous'] = {
                    'features': continuous_features,
                    'missing_before': 0,
                    'missing_after': 0,
                    'method': 'no_imputation_needed'
                }
        
        # Impute categorical features with mode
        if categorical_features:
            print(f"   🔄 Imputing {len(categorical_features)} categorical features with mode...")
            categorical_data = imputed_df[categorical_features]
            missing_before = categorical_data.isnull().sum().sum()
            
            if missing_before > 0:
                mode_imputer = SimpleImputer(strategy='most_frequent')
                imputed_categorical = mode_imputer.fit_transform(categorical_data)
                imputed_df[categorical_features] = imputed_categorical
                
                imputation_stats['categorical'] = {
                    'features': categorical_features,
                    'missing_before': int(missing_before),
                    'missing_after': int(imputed_df[categorical_features].isnull().sum().sum()),
                    'method': 'mode_imputation'
                }
            else:
                imputation_stats['categorical'] = {
                    'features': categorical_features,
                    'missing_before': 0,
                    'missing_after': 0,
                    'method': 'no_imputation_needed'
                }
        
        # Final validation - no missing values should remain
        final_missing = imputed_df[self.feature_mapper.all_features].isnull().sum().sum()
        if final_missing > 0:
            print(f"   ⚠️ Warning: {final_missing} missing values remain after imputation")
        else:
            print(f"   ✅ All missing values successfully imputed")
            
        return imputed_df, imputation_stats
        
    def create_enhanced_graph_data(self, dataset: pd.DataFrame) -> Data:
        """Create PyTorch Geometric data with enhanced 12 features."""
        print("\n🔄 Creating enhanced PyTorch Geometric graph data...")
        
        # Load production graph structure for consistency
        self.load_production_graph_data()
        
        # Extract enhanced features (12 features)
        feature_matrix = dataset[self.feature_mapper.all_features].values
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Create enhanced node features tensor
        x_enhanced = torch.FloatTensor(feature_matrix_scaled)
        
        # Use production model's graph structure (edges and labels)
        enhanced_graph_data = Data(
            x=x_enhanced,  # Enhanced 12 features instead of 7
            edge_index=self.production_graph_data.edge_index,  # Same graph structure
            edge_attr=self.production_graph_data.edge_attr,    # Same edge weights
            y=self.production_graph_data.y,                    # Same labels
            num_nodes=self.production_graph_data.num_nodes
        )
        
        # Add metadata
        enhanced_graph_data.feature_names = self.feature_mapper.all_features
        enhanced_graph_data.scaler = scaler
        enhanced_graph_data.version = "v1.1.0_enhanced"
        
        print(f"✅ Enhanced graph data created:")
        print(f"   📊 Nodes: {enhanced_graph_data.num_nodes}")
        print(f"   📊 Edges: {enhanced_graph_data.num_edges}")
        print(f"   📊 Features: {enhanced_graph_data.x.shape[1]} (enhanced from 7 to 12)")
        print(f"   📊 Feature names: {enhanced_graph_data.feature_names}")
        
        return enhanced_graph_data, scaler
        
    def save_enhanced_dataset(self, dataset: pd.DataFrame, metadata: Dict, 
                            graph_data: Data, scaler: StandardScaler) -> str:
        """Save complete enhanced dataset with all components."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save enhanced CSV dataset
        dataset_filename = f"enhanced_giman_12features_v1.1.0_{timestamp}.csv"
        dataset_path = self.output_dir / dataset_filename
        dataset.to_csv(dataset_path, index=False)
        
        # Save metadata
        metadata_filename = f"enhanced_metadata_v1.1.0_{timestamp}.json"
        metadata_path = self.output_dir / metadata_filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        # Save enhanced graph data
        graph_filename = f"enhanced_graph_data_v1.1.0_{timestamp}.pth"
        graph_path = self.output_dir / graph_filename
        torch.save(graph_data, graph_path)
        
        # Save scaler
        scaler_filename = f"enhanced_scaler_v1.1.0_{timestamp}.pkl"
        scaler_path = self.output_dir / scaler_filename
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        # Create latest symlinks
        latest_dataset = self.output_dir / "enhanced_dataset_latest.csv"
        latest_metadata = self.output_dir / "enhanced_metadata_latest.json"
        latest_graph = self.output_dir / "enhanced_graph_data_latest.pth"
        latest_scaler = self.output_dir / "enhanced_scaler_latest.pkl"
        
        # Remove existing symlinks
        for symlink in [latest_dataset, latest_metadata, latest_graph, latest_scaler]:
            if symlink.exists():
                symlink.unlink()
                
        # Create new symlinks
        latest_dataset.symlink_to(dataset_filename)
        latest_metadata.symlink_to(metadata_filename)
        latest_graph.symlink_to(graph_filename)
        latest_scaler.symlink_to(scaler_filename)
        
        print(f"\n💾 Enhanced dataset v1.1.0 saved:")
        print(f"   📄 Dataset: {dataset_path}")
        print(f"   📄 Metadata: {metadata_path}")
        print(f"   📄 Graph Data: {graph_path}")
        print(f"   📄 Scaler: {scaler_path}")
        print(f"   🔗 Latest symlinks created")
        
        return str(dataset_path)


def main():
    """Main execution function."""
    print("🚀 GIMAN Enhanced Dataset Creation v1.1.0")
    print("=" * 60)
    
    try:
        # Initialize creator
        creator = EnhancedDatasetCreator()
        
        # Create enhanced feature matrix
        enhanced_df, metadata = creator.create_enhanced_feature_matrix()
        
        # Impute missing values  
        imputed_df, imputation_stats = creator.impute_missing_values(enhanced_df)
        metadata['imputation_stats'] = imputation_stats
        
        # Create enhanced graph data
        enhanced_graph_data, scaler = creator.create_enhanced_graph_data(imputed_df)
        
        # Save complete enhanced dataset
        dataset_path = creator.save_enhanced_dataset(
            imputed_df, metadata, enhanced_graph_data, scaler)
        
        print(f"\n✅ Enhanced dataset creation complete!")
        print(f"📊 Features: 7 → 12 ({len(creator.feature_mapper.enhancement_features)} added)")
        print(f"📊 Dataset shape: {imputed_df.shape}")
        print(f"💾 Saved to: {dataset_path}")
        
        # Summary report
        print(f"\n📋 Enhancement Summary:")
        print(f"Current Features: {', '.join(creator.feature_mapper.current_features)}")
        print(f"Added Features: {', '.join(creator.feature_mapper.enhancement_features)}")
        
        # Feature coverage report
        print(f"\n📈 Feature Coverage Report:")
        for feature in creator.feature_mapper.all_features:
            if feature in metadata['feature_validation']:
                validation = metadata['feature_validation'][feature]
                status = "✅" if validation['coverage'] > 80 else "⚠️" if validation['coverage'] > 50 else "❌"
                print(f"   {status} {feature}: {validation['coverage']:.1f}%")
                
        return dataset_path
        
    except Exception as e:
        print(f"❌ Error creating enhanced dataset: {e}")
        raise


if __name__ == "__main__":
    main()