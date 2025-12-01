#!/usr/bin/env python
"""
Pipeline test script - validates the complete PPO training pipeline
with minimal compute (1-2 update cycles).
"""

import sys
from pathlib import Path
import yaml
from src.run import run

if __name__ == "__main__":
    print("=" * 60)
    print("PPO Pipeline Test")
    print("=" * 60)
    print("\nThis test will:")
    print("  1. Initialize MyFancyAgent")
    print("  2. Collect 64 steps of rollout data")
    print("  3. Compute GAE advantages")
    print("  4. Run 2 epochs of PPO updates")
    print("  5. Log all metrics to TensorBoard")
    print("  6. Save checkpoint")
    print("\nExpected runtime: 2-5 minutes")
    print("=" * 60)
    print()
    
    root_dir = Path(__file__).resolve().parent
    config_path = root_dir / 'configs' / 'test_params.yaml'
    
    # Load test configuration
    with config_path.open('r') as f:
        test_config = yaml.safe_load(f)
    
    print("Test Configuration:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Run the pipeline test
        run(test_config)
        print("\n" + "=" * 60)
        print("Pipeline test PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print("Pipeline test FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
