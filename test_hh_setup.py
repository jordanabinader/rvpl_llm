"""
Quick test to verify HH-RLHF setup is working
"""

import os
import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from hidden_context.data_utils.sequential_hh_dataset import SequentialHHDataset, sequential_hh_collate_fn
        from hidden_context.recurrent_vae_utils import RecurrentVAEModel
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_data_exists():
    """Test that required data files exist."""
    print("\nChecking data files...")
    
    # Check original data
    orig_path = "data/relabeled_hh_rlhf/both/train.jsonl.gz"
    if os.path.exists(orig_path):
        print(f"✓ Original data exists: {orig_path}")
    else:
        print(f"✗ Original data missing: {orig_path}")
        print("  This is expected - you have the original HH-RLHF data")
    
    # Check processed data
    proc_path = "data_release/hh_rlhf/gpt2/both/train.jsonl"
    if os.path.exists(proc_path):
        print(f"✓ Processed data exists: {proc_path}")
        
        # Count samples
        with open(proc_path, 'r') as f:
            num_samples = sum(1 for _ in f)
        print(f"  Found {num_samples} training samples")
        return True
    else:
        print(f"✗ Processed data missing: {proc_path}")
        print("  Run ./generate_hh_embeddings.sh to create it")
        return False


def test_dataset_loading():
    """Test that dataset can be loaded."""
    print("\nTesting dataset loading...")
    
    proc_path = "data_release/hh_rlhf/gpt2/both/train.jsonl"
    if not os.path.exists(proc_path):
        print("⊘ Skipping (processed data not found)")
        return False
    
    try:
        from hidden_context.data_utils.sequential_hh_dataset import SequentialHHDataset
        
        dataset = SequentialHHDataset(
            data_path="data_release/hh_rlhf/gpt2",
            data_subset="both",
            split="train",
            seq_length=10,
            epoch_size=100,
            seed=0
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Pool 0 (Helpful): {len(dataset.pools[0])} samples")
        print(f"  Pool 1 (Harmless): {len(dataset.pools[1])} samples")
        print(f"  Available user types: {dataset.available_user_types}")
        
        # Try loading one sample
        sample = dataset[0]
        print(f"✓ Sample loaded: {sample['embeddings_chosen'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("HH-RLHF Setup Validation")
    print("="*80)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Data Files", test_data_exists()))
    results.append(("Dataset Loading", test_dataset_loading()))
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ All tests passed! You're ready to train on HH-RLHF data.")
        print("\nNext steps:")
        print("  1. Run: ./run_streaming_vpl_hh.sh")
        print("  2. Check results in: experiments/streaming_vpl_hh/*/evaluation/")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nTo generate embeddings:")
        print("  1. Run: ./generate_hh_embeddings.sh")
        print("  2. Wait for completion (~10-30 minutes)")
        print("  3. Re-run this test script")
    print("="*80)


if __name__ == "__main__":
    main()

