#!/usr/bin/env python3
"""Quick test to check if all dependencies are working"""

print("Testing imports...")

try:
    print("1. Importing numpy...", end=" ", flush=True)
    import numpy as np
    print(f"✓ (version {np.__version__})")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

try:
    print("2. Importing pandas...", end=" ", flush=True)
    import pandas as pd
    print(f"✓ (version {pd.__version__})")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

try:
    print("3. Importing matplotlib...", end=" ", flush=True)
    import matplotlib
    print(f"✓ (version {matplotlib.__version__})")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

try:
    print("4. Importing scipy...", end=" ", flush=True)
    import scipy
    print(f"✓ (version {scipy.__version__})")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

try:
    print("5. Importing requests...", end=" ", flush=True)
    import requests
    print(f"✓ (version {requests.__version__})")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("\n✅ All dependencies working!")
print("\nRunning quick numpy test...")
arr = np.random.randn(5, 5)
print(f"Generated 5x5 random array: mean={arr.mean():.4f}, std={arr.std():.4f}")
print("\n🎉 Everything looks good! You can now run the main script.")
