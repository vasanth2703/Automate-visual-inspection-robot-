"""
Setup Supabase Storage Buckets
Creates the required storage buckets for the inspection system
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

print("="*70)
print("SUPABASE STORAGE SETUP")
print("="*70)

if not SUPABASE_URL or not SUPABASE_KEY:
    print("\n✗ Error: SUPABASE_URL and SUPABASE_KEY not found in .env")
    exit(1)

print(f"\nConnecting to: {SUPABASE_URL}")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Connected to Supabase")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

# Buckets to create
buckets = [
    {'name': 'scans', 'public': True},
    {'name': 'heatmaps', 'public': True},
    {'name': 'crops', 'public': True}
]

print("\nCreating storage buckets...")

for bucket_config in buckets:
    bucket_name = bucket_config['name']
    is_public = bucket_config['public']
    
    try:
        # Try to create bucket
        result = supabase.storage.create_bucket(
            bucket_name,
            options={'public': is_public}
        )
        print(f"  ✓ Created bucket: {bucket_name} (public: {is_public})")
    except Exception as e:
        error_msg = str(e)
        if 'already exists' in error_msg.lower():
            print(f"  ✓ Bucket already exists: {bucket_name}")
        else:
            print(f"  ✗ Failed to create {bucket_name}: {e}")

print("\n" + "="*70)
print("SETUP COMPLETE")
print("="*70)

# Verify buckets
print("\nVerifying buckets...")
try:
    buckets_list = supabase.storage.list_buckets()
    print(f"\nAvailable buckets:")
    for bucket in buckets_list:
        print(f"  - {bucket.name} (public: {bucket.public})")
except Exception as e:
    print(f"✗ Could not list buckets: {e}")

print("\n" + "="*70)
print("You can now run scans with image storage!")
print("="*70)
