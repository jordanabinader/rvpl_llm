"""
Simple test to verify PRISM dataset access.
"""

print("Testing PRISM dataset access...")
print()

# Test 1: Import datasets library
print("1. Testing datasets library import...")
try:
    from datasets import load_dataset
    print("   ✓ datasets library imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import datasets: {e}")
    print("   Install with: pip install datasets")
    exit(1)

# Test 2: Check HuggingFace connectivity
print()
print("2. Testing HuggingFace connectivity...")
try:
    # Try to load just the dataset config
    from datasets import get_dataset_config_names
    configs = get_dataset_config_names("HannahRoseKirk/prism-alignment")
    print(f"   ✓ Connected to HuggingFace")
    print(f"   Available configs: {configs}")
except Exception as e:
    print(f"   ✗ Failed to connect: {e}")
    print("   Check internet connection")
    exit(1)

# Test 3: Load a small sample
print()
print("3. Testing dataset loading (first 10 samples)...")
try:
    dataset = load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train[:10]")
    print(f"   ✓ Loaded {len(dataset)} samples")
    print(f"   Columns: {dataset.column_names}")
except Exception as e:
    print(f"   ✗ Failed to load dataset: {e}")
    exit(1)

# Test 4: Inspect structure
print()
print("4. Inspecting dataset structure...")
try:
    sample = dataset[0]
    print(f"   Sample keys: {list(sample.keys())[:10]}...")  # First 10 keys
    if 'conversation_id' in sample:
        print(f"   ✓ Has conversation_id: {sample['conversation_id']}")
    if 'user_id' in sample:
        print(f"   ✓ Has user_id: {sample['user_id'][:20]}...")
    if 'conversation_history' in sample:
        history = sample['conversation_history']
        print(f"   ✓ Has conversation_history with {len(history)} entries")
        if len(history) > 0:
            print(f"     First entry: {history[0]}")
except Exception as e:
    print(f"   ✗ Failed to inspect: {e}")
    exit(1)

print()
print("="*60)
print("✓ All tests passed!")
print("="*60)
print()
print("Next steps:")
print("  1. Review PRISM_SETUP.md for full documentation")
print("  2. Run: bash generate_prism_embeddings.sh")
print("  3. Run: bash run_streaming_vpl_prism.sh")

