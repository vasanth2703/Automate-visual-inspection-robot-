"""
Push to GitHub Script
Helps push the project to GitHub repository
"""
import subprocess
import os

REPO_URL = "https://github.com/vasanth2703/Automate-visual-inspection-robot-.git"

def run_command(cmd, description):
    """Run a git command"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} successful")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"✗ {description} failed")
            if result.stderr:
                print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("="*70)
    print("PUSH TO GITHUB")
    print("="*70)
    print(f"\nRepository: {REPO_URL}")
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("\nInitializing git repository...")
        run_command("git init", "Initialize git")
        run_command(f"git remote add origin {REPO_URL}", "Add remote")
    
    # Check git status
    print("\n" + "="*70)
    print("CURRENT STATUS")
    print("="*70)
    run_command("git status", "Check status")
    
    # Confirm
    print("\n" + "="*70)
    print("READY TO PUSH")
    print("="*70)
    print("\nThis will:")
    print("  1. Add all files (respecting .gitignore)")
    print("  2. Commit with message")
    print("  3. Push to GitHub")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        return
    
    # Get commit message
    commit_msg = input("\nCommit message (or press Enter for default): ").strip()
    if not commit_msg:
        commit_msg = "Complete YOLO + PatchCore Industrial Inspection System"
    
    # Add files
    if not run_command("git add .", "Add files"):
        return
    
    # Commit
    if not run_command(f'git commit -m "{commit_msg}"', "Commit changes"):
        print("\nNote: If 'nothing to commit', files are already committed")
    
    # Push
    print("\n" + "="*70)
    print("PUSHING TO GITHUB")
    print("="*70)
    
    # Try to push to main branch
    if not run_command("git push -u origin main", "Push to main"):
        # If main fails, try master
        print("\nTrying master branch...")
        run_command("git branch -M main", "Rename to main")
        run_command("git push -u origin main", "Push to main")
    
    print("\n" + "="*70)
    print("PUSH COMPLETE!")
    print("="*70)
    print(f"\nView your repository:")
    print(f"  {REPO_URL.replace('.git', '')}")
    print("\nNext steps:")
    print("  1. Go to GitHub repository")
    print("  2. Check files are uploaded")
    print("  3. Update README if needed")
    print("  4. Deploy using DEPLOYMENT.md guide")
    print("="*70)

if __name__ == '__main__':
    main()
