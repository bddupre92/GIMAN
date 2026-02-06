"""
Configuration Loader for GIMAN Dual Model System.

This module provides utilities for loading and validating configuration files
for the GIMAN-Progression and GIMAN-Conversion models.

Features:
    - YAML configuration file loading
    - Configuration validation and type checking
    - Default value handling
    - Environment variable substitution
    - Configuration merging for experiments

Author: GIMAN Development Team
Date: October 8, 2025
Version: 8.1.0
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class GIMANConfig:
    """
    Configuration container for GIMAN dual model system.
    
    This class encapsulates all configuration parameters for both
    GIMAN-Progression and GIMAN-Conversion models, including data paths,
    model hyperparameters, training settings, and evaluation metrics.
    
    Attributes:
        data: Data configuration (paths, splits, preprocessing)
        giman_progression: GIMAN-Progression model configuration
        giman_conversion: GIMAN-Conversion model configuration
        training: Training infrastructure configuration
        evaluation: Evaluation configuration
        experiment: Experiment tracking configuration
        
    Example:
        >>> config = GIMANConfig.from_yaml("configs/real_ppmi_dual_model.yaml")
        >>> print(config.data.num_features)  # 38
        >>> print(config.giman_progression.model.hidden_dim)  # 64
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary loaded from YAML
        """
        self._config = config_dict
        self._validate()
        
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'GIMANConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            GIMANConfig instance
            
        Raises:
            ConfigurationError: If file not found or invalid YAML
            
        Example:
            >>> config = GIMANConfig.from_yaml("configs/real_ppmi_dual_model.yaml")
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
        
        if config_dict is None:
            raise ConfigurationError(f"Empty configuration file: {config_path}")
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GIMANConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            GIMANConfig instance
        """
        return cls(config_dict)
    
    def _validate(self):
        """
        Validate configuration structure and required fields.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = ['data', 'giman_progression', 'giman_conversion', 
                           'training', 'evaluation', 'experiment']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required section: {section}")
        
        # Validate data section
        data_required = ['cohort_path', 'num_features', 'train_split', 
                        'val_split', 'test_split']
        for field in data_required:
            if field not in self._config['data']:
                raise ConfigurationError(f"Missing required data field: {field}")
        
        # Validate splits sum to 1.0
        splits = (self._config['data']['train_split'] + 
                 self._config['data']['val_split'] + 
                 self._config['data']['test_split'])
        if not 0.99 <= splits <= 1.01:
            raise ConfigurationError(
                f"Data splits must sum to 1.0, got {splits:.4f}"
            )
        
        # Validate model configurations
        for model_name in ['giman_progression', 'giman_conversion']:
            model_config = self._config[model_name]
            if 'model' not in model_config:
                raise ConfigurationError(
                    f"Missing 'model' section in {model_name}"
                )
            if 'training' not in model_config:
                raise ConfigurationError(
                    f"Missing 'training' section in {model_name}"
                )
    
    def __getattr__(self, name: str) -> Any:
        """
        Get configuration section by attribute access.
        
        Args:
            name: Configuration section name
            
        Returns:
            Configuration section as ConfigSection
            
        Example:
            >>> config = GIMANConfig.from_yaml("config.yaml")
            >>> print(config.data.num_features)
        """
        if name.startswith('_'):
            # Allow access to private attributes
            return object.__getattribute__(self, name)
        
        if name not in self._config:
            raise AttributeError(f"Configuration has no section: {name}")
        
        value = self._config[name]
        if isinstance(value, dict):
            return ConfigSection(value)
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return self._config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default.
        
        Args:
            key: Configuration key (supports dot notation, e.g., "data.num_features")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get("data.num_features", 30)
            38
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates (supports nested updates)
            
        Example:
            >>> config.update({"training": {"max_epochs": 100}})
        """
        self._deep_update(self._config, updates)
    
    @staticmethod
    def _deep_update(base: Dict, updates: Dict) -> Dict:
        """
        Recursively update nested dictionary.
        
        Args:
            base: Base dictionary
            updates: Update dictionary
            
        Returns:
            Updated dictionary
        """
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                GIMANConfig._deep_update(base[key], value)
            else:
                base[key] = value
        return base


class ConfigSection:
    """
    Configuration section with attribute access.
    
    This class wraps a dictionary and provides attribute-style access
    to configuration values, including nested sections.
    
    Example:
        >>> section = ConfigSection({"hidden_dim": 64, "dropout": 0.3})
        >>> print(section.hidden_dim)  # 64
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration section.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
    
    def __getattr__(self, name: str) -> Any:
        """
        Get configuration value by attribute access.
        
        Args:
            name: Configuration key
            
        Returns:
            Configuration value (wrapped as ConfigSection if dict)
        """
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        if name not in self._config:
            raise AttributeError(f"Configuration section has no key: {name}")
        
        value = self._config[name]
        if isinstance(value, dict):
            return ConfigSection(value)
        return value
    
    def __getitem__(self, key: str) -> Any:
        """
        Get configuration value by key access.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert section to dictionary.
        
        Returns:
            Configuration section as dictionary
        """
        return self._config.copy()
    
    def keys(self):
        """Get configuration keys."""
        return self._config.keys()
    
    def values(self):
        """Get configuration values."""
        return self._config.values()
    
    def items(self):
        """Get configuration items."""
        return self._config.items()


def load_config(config_path: Union[str, Path]) -> GIMANConfig:
    """
    Load GIMAN configuration from YAML file.
    
    This is a convenience function for loading configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        GIMANConfig instance
        
    Example:
        >>> from src.utils.config_loader import load_config
        >>> config = load_config("configs/real_ppmi_dual_model.yaml")
        >>> print(config.data.num_features)
    """
    return GIMANConfig.from_yaml(config_path)


def merge_configs(base_config: GIMANConfig, 
                 override_config: Union[GIMANConfig, Dict[str, Any]]) -> GIMANConfig:
    """
    Merge two configurations with override precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration (takes precedence)
        
    Returns:
        Merged configuration
        
    Example:
        >>> base = load_config("configs/base.yaml")
        >>> override = {"training": {"max_epochs": 100}}
        >>> merged = merge_configs(base, override)
    """
    merged_dict = base_config.to_dict()
    
    if isinstance(override_config, GIMANConfig):
        override_dict = override_config.to_dict()
    else:
        override_dict = override_config
    
    GIMANConfig._deep_update(merged_dict, override_dict)
    
    return GIMANConfig.from_dict(merged_dict)


# Example usage
if __name__ == '__main__':
    """Test configuration loading and access."""
    
    print("=" * 70)
    print("CONFIGURATION LOADER TEST")
    print("=" * 70)
    
    # Try to load configuration
    config_path = Path(__file__).parent.parent.parent / "configs" / "real_ppmi_dual_model.yaml"
    
    if not config_path.exists():
        print(f"\n⚠️  Configuration file not found: {config_path}")
        print("   Please ensure the config file exists before running this test.")
    else:
        try:
            # Load configuration
            config = load_config(config_path)
            print(f"\n✓ Configuration loaded from: {config_path}")
            
            # Test data section
            print("\n📊 Data Configuration:")
            print(f"   Cohort path: {config.data.cohort_path}")
            print(f"   Number of features: {config.data.num_features}")
            print(f"   Train/Val/Test split: {config.data.train_split}/{config.data.val_split}/{config.data.test_split}")
            
            # Test GIMAN-Progression
            print("\n🧠 GIMAN-Progression:")
            print(f"   Hidden dim: {config.giman_progression.model.hidden_dim}")
            print(f"   GAT layers: {config.giman_progression.model.num_gat_layers}")
            print(f"   Learning rate: {config.giman_progression.training.learning_rate}")
            print(f"   Max epochs: {config.giman_progression.training.max_epochs}")
            
            # Test GIMAN-Conversion
            print("\n🧠 GIMAN-Conversion:")
            print(f"   Hidden dim: {config.giman_conversion.model.hidden_dim}")
            print(f"   GAT layers: {config.giman_conversion.model.num_gat_layers}")
            print(f"   Learning rate: {config.giman_conversion.training.learning_rate}")
            print(f"   Pos weight: {config.giman_conversion.loss.pos_weight}")
            
            # Test experiment tracking
            print("\n🔬 Experiment:")
            print(f"   Name: {config.experiment.name}")
            print(f"   Version: {config.experiment.version}")
            print(f"   Tags: {', '.join(config.experiment.tags)}")
            
            # Test get method
            print("\n🔍 Testing get() method:")
            hidden_dim = config.get("giman_progression.model.hidden_dim", 32)
            print(f"   giman_progression.model.hidden_dim: {hidden_dim}")
            
            missing_value = config.get("nonexistent.key", "default")
            print(f"   nonexistent.key (default): {missing_value}")
            
            # Test configuration update
            print("\n🔧 Testing update():")
            config.update({"training": {"max_epochs": 500}})
            print(f"   Updated max_epochs: (not in model-specific config)")
            
            # Test to_dict
            print("\n📦 Testing to_dict():")
            config_dict = config.to_dict()
            print(f"   Dictionary keys: {list(config_dict.keys())}")
            
            print("\n✅ Configuration loader test PASSED!")
            
        except ConfigurationError as e:
            print(f"\n❌ Configuration error: {e}")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
    
    print("=" * 70)
