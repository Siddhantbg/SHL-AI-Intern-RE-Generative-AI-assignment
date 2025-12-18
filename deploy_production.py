#!/usr/bin/env python3
"""
Production deployment helper script
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False, e.stderr

def check_git_status():
    """Check if there are uncommitted changes."""
    success, output = run_command("git status --porcelain", "Checking git status")
    if success and output.strip():
        print("⚠️ You have uncommitted changes:")
        print(output)
        response = input("Do you want to commit and push them? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return commit_and_push()
        else:
            print("Please commit your changes before deploying.")
            return False
    return True

def commit_and_push():
    """Commit and push changes."""
    print("\n📝 Committing changes...")
    
    # Add all files
    success, _ = run_command("git add .", "Adding files to git")
    if not success:
        return False
    
    # Commit
    commit_msg = "Add production deployment configuration"
    success, _ = run_command(f'git commit -m "{commit_msg}"', "Committing changes")
    if not success:
        return False
    
    # Push
    success, _ = run_command("git push origin main", "Pushing to GitHub")
    return success

def display_deployment_options():
    """Display deployment options."""
    print("\n" + "="*60)
    print("🚀 PRODUCTION DEPLOYMENT OPTIONS")
    print("="*60)
    
    print("\n🚂 RAILWAY (Recommended - Easiest)")
    print("   1. Go to https://railway.app")
    print("   2. Sign up with GitHub")
    print("   3. Click 'New Project' → 'Deploy from GitHub repo'")
    print("   4. Select your repository")
    print("   5. Add environment variables:")
    print("      - ENVIRONMENT=production")
    print("      - DEBUG=false")
    print("      - PYTHONPATH=.")
    print("   6. Your app will be live at: https://your-app.railway.app")
    
    print("\n🎨 RENDER (Great Free Tier)")
    print("   1. Go to https://render.com")
    print("   2. Sign up with GitHub")
    print("   3. Click 'New +' → 'Web Service'")
    print("   4. Connect your repository")
    print("   5. Configure:")
    print("      - Build Command: pip install -r requirements.txt")
    print("      - Start Command: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT")
    print("   6. Add environment variables (same as Railway)")
    
    print("\n🟣 HEROKU (Classic)")
    print("   1. Install Heroku CLI")
    print("   2. Run: heroku create your-app-name")
    print("   3. Run: heroku config:set ENVIRONMENT=production DEBUG=false PYTHONPATH=.")
    print("   4. Run: git push heroku main")
    
    print("\n🌐 FRONTEND (Vercel)")
    print("   1. Go to https://vercel.com")
    print("   2. Import your GitHub repository")
    print("   3. Set Root Directory: frontend")
    print("   4. Add environment variable: REACT_APP_API_URL=https://your-api-url")

def main():
    """Main deployment function."""
    print("🎯 SHL Assessment Recommendation System - Production Deployment")
    print("="*70)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Not in project root directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check git status
    if not check_git_status():
        sys.exit(1)
    
    print("\n✅ Repository is ready for deployment!")
    
    # Display deployment options
    display_deployment_options()
    
    print("\n" + "="*60)
    print("🎉 READY TO DEPLOY!")
    print("="*60)
    print("Your repository now includes:")
    print("✅ railway.json - Railway deployment config")
    print("✅ render.yaml - Render deployment config") 
    print("✅ Procfile - Heroku deployment config")
    print("✅ runtime.txt - Python version specification")
    print("✅ .env.production - Production environment template")
    print("✅ Updated CORS configuration")
    print("✅ Production-ready requirements.txt")
    
    print("\n🚀 Choose your preferred platform and follow the steps above!")
    print("💡 I recommend Railway for the easiest deployment experience.")
    
    print("\n📚 For detailed instructions, see: PRODUCTION_DEPLOY.md")

if __name__ == "__main__":
    main()