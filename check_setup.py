"""
Setup Verification Script
Checks if the environment is properly configured
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        'torch', 'torchvision', 'transformers', 'diffusers',
        'accelerate', 'pillow', 'cv2', 'numpy', 'moviepy',
        'yaml', 'huggingface_hub'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    return len(missing) == 0

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU available: {device_name}")
            print(f"  Devices: {device_count}")
            print(f"  Memory: {total_memory:.1f} GB")
            return True
        else:
            print("⚠ GPU not available (CPU mode will be very slow)")
            return False
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    required_dirs = [
        'configs', 'scripts', 'utils', 'outputs'
    ]
    
    required_files = [
        'main.py', 'requirements.txt', 'configs/model_config.yaml',
        'scripts/sample_script.txt'
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ Directory: {dir_name}/")
        else:
            print(f"❌ Missing directory: {dir_name}/")
            all_ok = False
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✓ File: {file_name}")
        else:
            print(f"❌ Missing file: {file_name}")
            all_ok = False
    
    return all_ok

def check_skyreels_integration():
    """Check if SkyReels-V2 is integrated"""
    skyreels_paths = [
        Path("./skyreels_v2_infer"),
        Path("../SkyReels-V2/skyreels_v2_infer"),
        Path("./SkyReels-V2/skyreels_v2_infer")
    ]
    
    for path in skyreels_paths:
        if path.exists():
            print(f"✓ SkyReels-V2 found at: {path}")
            return True
    
    print("⚠ SkyReels-V2 not found - integration required")
    print("  See INTEGRATION_NOTES.md for integration steps")
    return False

def main():
    print("=" * 60)
    print("SkyReels Movie Generator - Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("GPU", check_gpu),
        ("Project Structure", check_project_structure),
        ("SkyReels Integration", check_skyreels_integration),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        print("-" * 40)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All checks passed! Ready to use.")
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        print("  See QUICKSTART.md and INTEGRATION_NOTES.md for help.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

