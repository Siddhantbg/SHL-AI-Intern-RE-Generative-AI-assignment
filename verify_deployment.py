#!/usr/bin/env python3
"""
Deployment verification script to ensure everything is set up correctly.
"""

import os
import sys
import importlib.util

def check_app_import():
    """Check if app.py can be imported successfully."""
    try:
        import app
        print("✅ app.py imports successfully")
        print(f"✅ FastAPI app created: {type(app.app)}")
        print(f"✅ Routes available: {len([r for r in app.app.routes])}")
        return True
    except Exception as e:
        print(f"❌ Failed to import app.py: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available."""
    required_deps = ['fastapi', 'uvicorn', 'google.generativeai']
    
    for dep in required_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"❌ {dep} is missing")
            return False
    return True

def check_files():
    """Check if all required files exist."""
    required_files = ['app.py', 'Procfile', 'requirements.txt']
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} is missing")
            return False
    return True

def check_procfile():
    """Check Procfile content."""
    try:
        with open('Procfile', 'r') as f:
            content = f.read().strip()
        
        if 'app:app' in content:
            print(f"✅ Procfile correctly points to app:app")
            print(f"   Content: {content}")
        else:
            print(f"❌ Procfile doesn't point to app:app")
            print(f"   Content: {content}")
            return False
        return True
    except Exception as e:
        print(f"❌ Failed to read Procfile: {e}")
        return False

def main():
    print("🔍 Verifying deployment setup...\n")
    
    checks = [
        ("Files", check_files),
        ("Dependencies", check_dependencies), 
        ("App Import", check_app_import),
        ("Procfile", check_procfile)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 Checking {name}:")
        if not check_func():
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All checks passed! Ready for deployment.")
        print("\n📝 Next steps:")
        print("1. Commit and push changes to GitHub")
        print("2. In Render dashboard, trigger a manual deploy")
        print("3. Check deployment logs for any issues")
    else:
        print("❌ Some checks failed. Fix issues before deploying.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())