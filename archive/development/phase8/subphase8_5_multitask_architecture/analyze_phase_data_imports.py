"""
Analyze Data Imports Across Phases 1-8

This script systematically analyzes all Python files in Phases 1-8 to identify:
1. What data files are being loaded (CSV, JSON, PTH, NPZ, etc.)
2. What columns/features are being used
3. Data transformations and preprocessing
4. Model architectures and their I/O dimensions
5. Cross-phase data dependencies

Purpose: Ensure Phase 8.5 multi-task architecture uses correct data sources
"""

import re
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Set, Tuple

class PhaseDataAnalyzer:
    """Analyze data usage patterns across GIMAN phases."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.data_files = defaultdict(list)  # file -> phases using it
        self.model_architectures = defaultdict(dict)  # phase -> architecture info
        self.feature_usage = defaultdict(set)  # phase -> feature columns
        self.cross_references = defaultdict(list)  # file -> dependencies
        
    def analyze_all_phases(self):
        """Analyze all phases 1-8."""
        print("=" * 80)
        print("GIMAN PHASE DATA IMPORT ANALYSIS")
        print("=" * 80)
        
        for phase_num in range(1, 9):
            phase_dir = self.base_path / f"phase{phase_num}"
            if phase_dir.exists():
                print(f"\n{'='*80}")
                print(f"ANALYZING PHASE {phase_num}")
                print(f"{'='*80}")
                self.analyze_phase(phase_num, phase_dir)
        
        # Special handling for Phase 8 subphases
        phase8_dir = self.base_path / "phase8"
        if phase8_dir.exists():
            for subphase_dir in phase8_dir.glob("subphase8_*"):
                subphase_num = subphase_dir.name.split("_")[1]
                print(f"\n{'='*80}")
                print(f"ANALYZING PHASE 8.{subphase_num}")
                print(f"{'='*80}")
                self.analyze_phase(f"8.{subphase_num}", subphase_dir)
        
        # Generate summary report
        self.generate_summary()
    
    def analyze_phase(self, phase_id: str, phase_dir: Path):
        """Analyze a single phase directory."""
        python_files = list(phase_dir.glob("**/*.py"))
        
        if not python_files:
            print(f"  No Python files found in {phase_dir.name}")
            return
        
        print(f"\n  Found {len(python_files)} Python files")
        
        for py_file in python_files:
            try:
                self.analyze_file(phase_id, py_file)
            except Exception as e:
                print(f"  WARNING: Error analyzing {py_file.name}: {e}")
    
    def analyze_file(self, phase_id: str, file_path: Path):
        """Analyze a single Python file for data usage."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return
        
        # Skip if file is too small (likely template)
        if len(content) < 100:
            return
        
        # Extract data file references
        data_patterns = [
            r'pd\.read_csv\([\'"]([^\'"]+ \.csv)[\'"]',
            r'\.read_csv\([\'"]([^\'"]+ \.csv)[\'"]',
            r'json\.load\([\'"]([^\'"]+ \.json)[\'"]',
            r'torch\.load\([\'"]([^\'"]+ \.pth)[\'"]',
            r'torch\.load\([\'"]([^\'"]+ \.pt)[\'"]',
            r'np\.load\([\'"]([^\'"]+ \.npz)[\'"]',
            r'[\'"]data/[^\'"]+ ',
            r'[\'"]outputs/[^\'"]+ ',
            r'[\'"]models/[^\'"]+ ',
        ]
        
        found_data = False
        for pattern in data_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if not found_data:
                    print(f"\n  FILE: {file_path.name}")
                    found_data = True
                print(f"     -> {match}")
                self.data_files[match].append(f"Phase {phase_id}")
        
        # Extract feature/column usage
        column_patterns = [
            r'\[[\'"](PATNO|EVENT_ID|COHORT|phenoconverted|SAA_POSITIVE|time_to_event|event_observed)[\'"]\ ]',
            r'\.columns\s*=\s*\[([^\]]+)\]',
            r'features\s*=\s*\[([^\]]+)\]',
        ]
        
        for pattern in column_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match:
                    self.feature_usage[phase_id].add(match)
        
        # Extract model architecture info
        if "class" in content and ("nn.Module" in content or "Module" in content):
            class_matches = re.findall(r'class\s+(\w+)\s*\([^)]*Module[^)]*\)', content)
            for class_name in class_matches:
                if class_name not in self.model_architectures[phase_id]:
                    print(f"     -> MODEL: {class_name}")
                    self.model_architectures[phase_id][class_name] = {
                        'file': file_path.name,
                        'phase': phase_id
                    }
                    
                    # Extract input/output dimensions
                    init_match = re.search(
                        rf'class\s+{class_name}.*?def\s+__init__\s*\([^)]*\):.*?(?=\n    def|\nclass|\Z)',
                        content,
                        re.DOTALL
                    )
                    if init_match:
                        init_content = init_match.group(0)
                        # Look for input_dim, hidden_dim, output_dim
                        dims = {}
                        for dim_type in ['input_dim', 'in_features', 'hidden_dim', 'output_dim', 'num_classes']:
                            dim_match = re.search(rf'{dim_type}[:\s=]+(\d+)', init_content)
                            if dim_match:
                                dims[dim_type] = int(dim_match.group(1))
                        
                        if dims:
                            self.model_architectures[phase_id][class_name]['dimensions'] = dims
                            print(f"        -> Dimensions: {dims}")
    
    def generate_summary(self):
        """Generate comprehensive summary report."""
        print("\n" + "=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        
        # Data files used across phases
        print("\nDATA FILES USAGE ACROSS PHASES")
        print("-" * 80)
        for data_file, phases in sorted(self.data_files.items()):
            print(f"\n  {data_file}")
            print(f"    Used in: {', '.join(set(phases))}")
        
        # Model architectures by phase
        print("\n\nMODEL ARCHITECTURES BY PHASE")
        print("-" * 80)
        # Sort by converting to string for mixed int/str keys
        sorted_phases = sorted(self.model_architectures.items(), key=lambda x: str(x[0]))
        for phase_id, models in sorted_phases:
            print(f"\n  Phase {phase_id}:")
            for model_name, info in models.items():
                print(f"    ├─ {model_name} ({info.get('file', 'unknown')})")
                if 'dimensions' in info:
                    for dim_name, dim_val in info['dimensions'].items():
                        print(f"    │  └─ {dim_name}: {dim_val}")
        
        # Critical findings for Phase 8.5
        print("\n\nPHASE 8.5 MULTI-TASK ARCHITECTURE REQUIREMENTS")
        print("-" * 80)
        
        # Task 1: Progression (from Phase 8.2)
        print("\n  Task 1: Progression Prediction")
        prog_files = [f for f in self.data_files.keys() if 'progression' in f.lower() or 'survival' in f.lower()]
        if prog_files:
            print(f"    OK: Data available: {prog_files}")
        else:
            print(f"    MISSING: No progression data found")
        
        # Task 2: Conversion (phenoconversion)
        print("\n  Task 2: Phenoconversion Prediction")
        conv_files = [f for f in self.data_files.keys() if 'conversion' in f.lower() or 'phenoconver' in f.lower()]
        if conv_files:
            print(f"    OK: Data available: {conv_files}")
        else:
            print(f"    MISSING: No conversion data found")
        
        # Task 3: SAA (from Phase 8.3)
        print("\n  Task 3: SAA Prediction")
        saa_files = [f for f in self.data_files.keys() if 'saa' in f.lower()]
        if saa_files:
            print(f"    OK: Data available: {saa_files}")
        else:
            print(f"    MISSING: No SAA data found")
        
        # Task 4: Diagnostic (PD/Prodromal/Control)
        print("\n  Task 4: Diagnostic Classification")
        diag_files = [f for f in self.data_files.keys() if 'cohort' in f.lower() or 'diagnostic' in f.lower()]
        if diag_files:
            print(f"    OK: Potential data: {diag_files}")
        else:
            print(f"    NOTE: Need to extract from COHORT column")
        
        # Trained models available for reuse
        print("\n\nTRAINED MODELS AVAILABLE FOR PHASE 8.5")
        print("-" * 80)
        
        print("\n  From Phase 6 (GAT Backbone):")
        phase6_models = self.model_architectures.get('6', {})
        if 'GIMANBackboneGAT' in phase6_models:
            print("    OK: GIMANBackboneGAT - Shared encoder")
        else:
            print("    NOTE: Need to locate GIMANBackboneGAT")
        
        print("\n  From Phase 5 (Survival Analysis):")
        phase5_models = self.model_architectures.get('5', {})
        if 'DeepSurv' in phase5_models:
            print("    OK: DeepSurv - Survival prediction head")
        else:
            print("    NOTE: Need to locate DeepSurv")
        
        print("\n  From Phase 8.2 (Progression):")
        phase82_models = self.model_architectures.get('8.2', {})
        if phase82_models:
            print(f"    OK: Found {len(phase82_models)} model(s):")
            for model_name in phase82_models:
                print(f"       -> {model_name}")
        else:
            print("    NOTE: No models found in Phase 8.2")
        
        print("\n  From Phase 8.3 (SAA):")
        phase83_models = self.model_architectures.get('8.3', {})
        if phase83_models:
            print(f"    OK: Found {len(phase83_models)} model(s):")
            for model_name in phase83_models:
                print(f"       -> {model_name}")
        else:
            print("    NOTE: No models found in Phase 8.3")
        
        # Save to JSON
        output = {
            'data_files': dict(self.data_files),
            'model_architectures': dict(self.model_architectures),
            'feature_usage': {k: list(v) for k, v in self.feature_usage.items()}
        }
        
        output_file = Path(__file__).parent / "phase_data_analysis_report.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n\nFull report saved to: {output_file}")


def main():
    """Main execution function."""
    # Get base path (development directory)
    script_path = Path(__file__).resolve()
    dev_path = script_path.parents[2]  # Go up to development/
    
    print(f"Analyzing phases in: {dev_path}\n")
    
    analyzer = PhaseDataAnalyzer(dev_path)
    analyzer.analyze_all_phases()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
