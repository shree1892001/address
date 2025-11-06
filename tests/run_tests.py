#!/usr/bin/env python3
"""
Test runner script for OCR Table Extraction API tests
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests(test_type="all", verbose=False, coverage=False, parallel=False):
    """
    Run the test suite with specified options
    
    Args:
        test_type (str): Type of tests to run ('all', 'unit', 'integration', 'api')
        verbose (bool): Run tests in verbose mode
        coverage (bool): Run tests with coverage reporting
        parallel (bool): Run tests in parallel
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    cmd.append("tests/")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.append("--cov=app")
        cmd.append("--cov-report=html")
        cmd.append("--cov-report=term-missing")
    
    # Add parallel execution if requested
    if parallel:
        cmd.append("-n")
        cmd.append("auto")
    
    # Add test type filters
    if test_type == "unit":
        cmd.append("-m")
        cmd.append("unit")
    elif test_type == "integration":
        cmd.append("-m")
        cmd.append("integration")
    elif test_type == "api":
        cmd.append("-m")
        cmd.append("api")
    
    # Add additional options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings"  # Disable warnings for cleaner output
    ])
    
    print(f"Running tests with command: {' '.join(cmd)}")
    print(f"Test type: {test_type}")
    print(f"Verbose: {verbose}")
    print(f"Coverage: {coverage}")
    print(f"Parallel: {parallel}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 50)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"❌ Tests failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_specific_test(test_file, verbose=False):
    """
    Run a specific test file
    
    Args:
        test_file (str): Path to the test file
        verbose (bool): Run tests in verbose mode
    """
    cmd = ["python", "-m", "pytest", test_file]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])
    
    print(f"Running specific test: {test_file}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 50)
        print("✅ Test passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"❌ Test failed with exit code: {e.returncode}")
        return False


def check_test_environment():
    """Check if the test environment is properly set up"""
    print("🔍 Checking test environment...")
    
    # Check if required packages are installed
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-xdist",
        "fastapi",
        "httpx"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    # Check if test directories exist
    test_dirs = ["tests", "app"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"✅ {test_dir}/")
        else:
            print(f"❌ {test_dir}/ - NOT FOUND")
            return False
    
    print("✅ Test environment is ready!")
    return True


def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Run OCR Table Extraction API tests")
    
    parser.add_argument(
        "--type", "-t",
        choices=["all", "unit", "integration", "api"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests in verbose mode"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage reporting"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--file", "-f",
        help="Run a specific test file"
    )
    
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check test environment setup"
    )
    
    args = parser.parse_args()
    
    if args.check_env:
        success = check_test_environment()
        sys.exit(0 if success else 1)
    
    if args.file:
        success = run_specific_test(args.file, args.verbose)
    else:
        success = run_tests(args.type, args.verbose, args.coverage, args.parallel)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
