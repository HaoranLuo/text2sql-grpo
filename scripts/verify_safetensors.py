import sys
from safetensors import safe_open

for path in sys.argv[1:]:
    try:
        with safe_open(path, framework="pt") as f:
            print(f"OK: {path.split('/')[-1]} tensors={len(f.keys())}")
    except Exception as e:
        print(f"FAIL: {path} -> {e}")
        sys.exit(1)
