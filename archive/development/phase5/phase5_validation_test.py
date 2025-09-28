#!/usr/bin/env python3
"""GIMAN Phase 5: Quick Validation Test.

Simple validation test to ensure Phase 5 systems can be imported and 
basic functionality works before running full experiments.

This test validates:
1. Import functionality for all Phase 5 modules
2. Basic model instantiation
3. Data flow through architectures
4. Integration with Phase 3 data system

Author: GIMAN Development Team  
Date: September 2025
"""

import logging
import warnings
import sys
from pathlib import Path

import numpy as np
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add path for phase3 integration
sys.path.append(
    "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase3"
)

def test_phase5_imports():
    """Test Phase 5 module imports."""
    logger.info("🔍 Testing Phase 5 imports...")
    
    try:
        # Test Phase 3 integration
        from phase3.phase3_1_real_data_integration import RealDataPhase3Integration
        logger.info("   ✅ Phase 3 integration import successful")
        
        # Test Phase 5 task-specific architecture
        sys.path.append(
            "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase5"
        )
        
        from phase5_task_specific_giman import TaskSpecificGIMANSystem, TaskSpecificLOOCVEvaluator
        logger.info("   ✅ Task-specific GIMAN import successful")
        
        from phase5_dynamic_loss_system import DynamicLossWeighter, DynamicLossGIMANSystem
        logger.info("   ✅ Dynamic loss system import successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"   ❌ Import failed: {e}")
        return False

def test_model_instantiation():
    """Test basic model instantiation."""
    logger.info("⚙️ Testing model instantiation...")
    
    try:
        from phase5_task_specific_giman import TaskSpecificGIMANSystem
        from phase5_dynamic_loss_system import DynamicLossGIMANSystem, DynamicLossWeighter
        
        # Test task-specific model
        model1 = TaskSpecificGIMANSystem(
            spatial_dim=256,
            genomic_dim=8, 
            temporal_dim=64,
            embed_dim=32
        )
        logger.info("   ✅ Task-specific GIMAN instantiation successful")
        
        # Test dynamic loss model
        model2 = DynamicLossGIMANSystem(
            loss_strategy="fixed",
            spatial_dim=256,
            genomic_dim=8,
            temporal_dim=64,
            embed_dim=32
        )
        logger.info("   ✅ Dynamic loss GIMAN instantiation successful")
        
        # Test loss weighter
        weighter = DynamicLossWeighter(strategy="adaptive")
        logger.info("   ✅ Dynamic loss weighter instantiation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Model instantiation failed: {e}")
        return False

def test_data_integration():
    """Test data integration with Phase 3 system."""
    logger.info("📊 Testing data integration...")
    
    try:
        from phase3.phase3_1_real_data_integration import RealDataPhase3Integration
        
        # Initialize integrator
        integrator = RealDataPhase3Integration()
        logger.info("   ✅ Phase 3 integrator initialized")
        
        # Load embeddings
        integrator.load_and_validate_embeddings()
        logger.info("   ✅ Embeddings loaded and validated")
        
        # Create adjacency matrices
        integrator.create_adjacency_matrices()
        logger.info("   ✅ Adjacency matrices created")
        
        # Generate targets
        integrator.generate_prognostic_targets()
        logger.info("   ✅ Prognostic targets generated")
        
        # Check data shapes
        spatial_shape = integrator.spatiotemporal_embeddings.shape
        genomic_shape = integrator.genomic_embeddings.shape
        temporal_shape = integrator.temporal_embeddings.shape
        
        logger.info(f"   📈 Data shapes:")
        logger.info(f"      Spatial: {spatial_shape}")
        logger.info(f"      Genomic: {genomic_shape}")
        logger.info(f"      Temporal: {temporal_shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Data integration failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass through Phase 5 models."""
    logger.info("🔄 Testing forward pass...")
    
    try:
        from phase3.phase3_1_real_data_integration import RealDataPhase3Integration
        from phase5_task_specific_giman import TaskSpecificGIMANSystem
        from phase5_dynamic_loss_system import DynamicLossGIMANSystem
        
        # Load data
        integrator = RealDataPhase3Integration()
        integrator.load_and_validate_embeddings()
        integrator.create_adjacency_matrices()
        integrator.generate_prognostic_targets()
        
        # Get sample data (first patient)
        spatial = torch.FloatTensor(integrator.spatiotemporal_embeddings[:1])
        genomic = torch.FloatTensor(integrator.genomic_embeddings[:1])
        temporal = torch.FloatTensor(integrator.temporal_embeddings[:1])
        adj_matrix = torch.FloatTensor(integrator.adjacency_matrices[0])
        
        # Test task-specific model
        model1 = TaskSpecificGIMANSystem()
        motor_pred1, cognitive_pred1 = model1(spatial, genomic, temporal, adj_matrix)
        logger.info(f"   ✅ Task-specific forward pass: motor={motor_pred1.shape}, cognitive={cognitive_pred1.shape}")
        
        # Test dynamic loss model
        model2 = DynamicLossGIMANSystem(loss_strategy="fixed")
        motor_pred2, cognitive_pred2 = model2(spatial, genomic, temporal, adj_matrix)
        logger.info(f"   ✅ Dynamic loss forward pass: motor={motor_pred2.shape}, cognitive={cognitive_pred2.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Forward pass failed: {e}")
        return False

def test_loss_weighting():
    """Test dynamic loss weighting functionality."""
    logger.info("⚖️ Testing loss weighting...")
    
    try:
        from phase5_dynamic_loss_system import DynamicLossWeighter
        
        # Test different strategies
        strategies = ["fixed", "adaptive", "curriculum"]
        
        for strategy in strategies:
            weighter = DynamicLossWeighter(strategy=strategy)
            
            # Test weight calculation
            motor_weight, cognitive_weight = weighter.get_weights(0.5, 0.3, epoch=10)
            
            logger.info(f"   ✅ {strategy.upper()}: motor={motor_weight:.3f}, cognitive={cognitive_weight:.3f}")
            
            # Validate weights sum to reasonable range
            total_weight = motor_weight + cognitive_weight
            assert 0.8 <= total_weight <= 1.2, f"Weights sum to {total_weight}, expected ~1.0"
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Loss weighting failed: {e}")
        return False

def run_phase5_validation():
    """Run comprehensive Phase 5 validation tests."""
    logger.info("🚀 Starting Phase 5 Validation Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Import Functionality", test_phase5_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Data Integration", test_data_integration),
        ("Forward Pass", test_forward_pass),
        ("Loss Weighting", test_loss_weighting),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} Test")
        logger.info("-" * 30)
        
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name} Test: PASSED")
            else:
                logger.error(f"❌ {test_name} Test: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} Test: FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n📋 VALIDATION SUMMARY")
    logger.info("=" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 5 validation tests PASSED!")
        logger.info("   Phase 5 system is ready for experiments")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests FAILED")
        logger.warning("   Phase 5 system may have issues")
        return False


if __name__ == "__main__":
    success = run_phase5_validation()
    
    if success:
        print("\n🎉 Phase 5 Validation Complete!")
        print("✅ All systems operational")
        print("🚀 Ready for Phase 5 experiments")
    else:
        print("\n⚠️ Phase 5 Validation Issues Detected")
        print("❌ Some systems may not be operational")
        print("🔧 Review logs for specific issues")