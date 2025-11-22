# Supabase Setup Guide

## Quick Setup (5 minutes)

### 1. Database Schema
```bash
# Copy content of supabase_schema.sql
# Paste into Supabase SQL Editor
# Click "Run"
```

### 2. Storage Buckets (Optional but Recommended)
```bash
py -3.12 setup_supabase_storage.py
```

This creates 3 storage buckets:
- `scans` - For camera images
- `heatmaps` - For anomaly heatmaps  
- `crops` - For component crops

**Note:** System works without storage buckets! Data is still saved to database, but images won't be uploaded to cloud storage.

### 3. Test Connection
```bash
py -3.12 test_supabase.py
```

### 4. Verify Setup
```bash
py -3.12 test_realtime_flow.py
```

## What Gets Saved

### With Storage Buckets:
✅ Scan records
✅ Frame records with image URLs
✅ Detection records
✅ Images in cloud storage
✅ Component statistics

### Without Storage Buckets:
✅ Scan records
✅ Frame records (no image URLs)
✅ Detection records
✅ Component statistics
⚠️ Images saved locally only

## Troubleshooting

### "Bucket not found" errors
- Run: `py -3.12 setup_supabase_storage.py`
- Or create buckets manually in Supabase Dashboard

### No data in database
- Check SQL schema is created
- Verify .env credentials
- Run: `py -3.12 test_supabase.py`

### Dashboard not updating
- Check browser console for errors
- Verify Supabase URL in react_dashboard.html
- Check real-time subscriptions are enabled in Supabase

## Manual Bucket Creation

If automatic setup fails, create manually:

1. Go to Supabase Dashboard > Storage
2. Click "New bucket"
3. Name: `scans`, Public: Yes
4. Repeat for `heatmaps` and `crops`

## Success Indicators

✅ `py -3.12 test_supabase.py` - All tests pass
✅ `py -3.12 test_realtime_flow.py` - Scan completes
✅ Dashboard shows live data
✅ Supabase dashboard shows records

## Next Steps

After setup:
1. Start backend: `py -3.12 run_server.py`
2. Run scan: Visit http://localhost:8000/docs
3. Open dashboard: `py -3.12 launch_dashboard.py`
