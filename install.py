#!/usr/bin/env python
"""
ComfyUI-ZSNodes Installation Script
Automatically installs required dependencies
"""

import subprocess
import sys
import os
import importlib.util
from typing import List, Tuple

def check_package_installed(package_name: str) -> bool:
    """Check if a package is already installed"""
    spec = importlib.util.find_spec(package_name.split('[')[0].split('@')[0].split('>')[0].split('=')[0])
    return spec is not None

def install_package(package: str) -> bool:
    """Install a single package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.STDOUT)
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def get_requirements() -> Tuple[List[str], List[str]]:
    """Get lists of required and optional packages"""
    required = [
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
    ]
    
    optional = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "safetensors>=0.3.1",
    ]
    
    return required, optional

def main():
    print("=" * 60)
    print("ComfyUI-ZSNodes Installation")
    print("=" * 60)
    
    # Get requirements
    required_packages, optional_packages = get_requirements()
    
    # Install required packages
    print("\n📦 Installing required packages...")
    failed_required = []
    for package in required_packages:
        package_name = package.split('>')[0].split('=')[0]
        if check_package_installed(package_name):
            print(f"✓ {package_name} already installed")
        else:
            if not install_package(package):
                failed_required.append(package)
    
    # Install optional packages (for Bounding Box Crop node)
    print("\n📦 Installing optional packages (for Bounding Box Crop node)...")
    failed_optional = []
    
    for package in optional_packages:
        package_name = package.split('>')[0].split('=')[0]
        if check_package_installed(package_name):
            print(f"✓ {package_name} already installed")
        else:
            if not install_package(package):
                failed_optional.append(package)
    
    # Special handling for GroundingDINO
    print("\n📦 Checking GroundingDINO installation...")
    if check_package_installed("groundingdino"):
        print("✓ GroundingDINO already installed")
    else:
        print("Installing GroundingDINO (this may take a while)...")
        try:
            # First, ensure build dependencies are installed
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wheel", "setuptools"],
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.STDOUT)
            
            # Install GroundingDINO from GitHub
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/IDEA-Research/GroundingDINO.git"
            ])
            print("✓ Successfully installed GroundingDINO")
        except subprocess.CalledProcessError as e:
            print(f"⚠ GroundingDINO installation failed: {e}")
            print("  The Bounding Box Crop node will not be available.")
            print("  You can try installing it manually later with:")
            print("  pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
            failed_optional.append("groundingdino")
    
    # Summary
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    
    if not failed_required:
        print("✓ All required packages installed successfully")
        print("  - Save Image (ZS Custom) node is ready to use")
    else:
        print("✗ Some required packages failed to install:")
        for package in failed_required:
            print(f"  - {package}")
        print("\nPlease install them manually with:")
        print(f"pip install {' '.join(failed_required)}")
    
    if not failed_optional:
        if "groundingdino" not in failed_optional:
            print("✓ All optional packages installed successfully")
            print("  - Bounding Box Crop (ZS) node is ready to use")
    else:
        print("\n⚠ Some optional packages failed to install:")
        for package in failed_optional:
            print(f"  - {package}")
        print("\nThe Bounding Box Crop node may not be available.")
        print("You can install these manually if needed.")
    
    print("\n" + "=" * 60)
    print("Installation complete! Please restart ComfyUI.")
    print("=" * 60)
    
    # Return 0 for success, 1 if required packages failed
    return 1 if failed_required else 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Installation failed with error: {e}")
        sys.exit(1)