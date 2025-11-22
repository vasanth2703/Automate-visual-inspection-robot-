# Quick Fix Guide

## Issue: Database Constraint Error

**Error:** `detections_patchcore_score_check` violation

**Cause:** PatchCore scores can exceed 1.0, but database expects 0-1 range

## Fix (Choose One):

### Option A: Update Database (Recommended)
```sql
-- Run in Supabase SQL Editor
ALTER TABLE detections DROP CONSTRAINT IF EXISTS detections_patchcore_score_check;
ALTER TABLE detections ADD CONSTRAINT detections_patchcore_score_check CHECK (patchcore_score >= 0);
```

Or use the provided file:
```bash
# Copy content of fix_database_constraint.sql
# Paste into Supabase SQL Editor
# Click "Run"
```

### Option B: Recreate Database
```bash
# Use updated schema
# Copy content of supabase_schema.sql
# Paste into Supabase SQL Editor
# Click "Run"
```

## Verify Fix

```bash
py -3.12 test_complete_integration.py
```

Should show:
```
✓ PASS - backend
✓ PASS - ai
✓ PASS - database
✓ PASS - scan
✓ PASS - results
✓ PASS - dashboard
```

## Complete System Test

```bash
# Terminal 1: Backend
py -3.12 run_server.py

# Terminal 2: Test
py -3.12 test_complete_integration.py

# Terminal 3: Dashboard
py -3.12 launch_dashboard.py
```

## What's Fixed

✅ PatchCore scores normalized (0-1 range)
✅ Database constraint updated
✅ Real-time dashboard updates
✅ Camera feed displays
✅ Memory issues resolved
✅ All systems integrated

## Dashboard Features

- **Auto-refresh**: Updates without manual refresh
- **Live feed**: Shows camera images in real-time
- **Notifications**: Alerts for scan events
- **Connection status**: Green = connected
- **Real-time stats**: Updates as data arrives

## Troubleshooting

### Still seeing errors?
1. Clear browser cache
2. Restart backend
3. Check Supabase dashboard for data
4. Run: `py -3.12 test_supabase.py`

### Dashboard not updating?
1. Check browser console (F12)
2. Look for "Subscription status: SUBSCRIBED"
3. Verify green connection indicator
4. Check Supabase real-time is enabled

### No images in dashboard?
1. Run: `py -3.12 setup_supabase_storage.py`
2. Check storage buckets exist
3. Images still saved locally in data/results/

## Success Indicators

✅ Green "Connected" in dashboard header
✅ "Real-time updates active" notification
✅ Camera feed shows images
✅ Scan list updates automatically
✅ No manual refresh needed
