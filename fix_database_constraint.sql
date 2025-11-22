-- Fix PatchCore Score Constraint
-- Run this in Supabase SQL Editor to fix the constraint

-- Drop the old constraint
ALTER TABLE detections DROP CONSTRAINT IF EXISTS detections_patchcore_score_check;

-- Add new constraint (no upper limit)
ALTER TABLE detections ADD CONSTRAINT detections_patchcore_score_check 
CHECK (patchcore_score >= 0);

-- Verify the change
SELECT conname, pg_get_constraintdef(oid) 
FROM pg_constraint 
WHERE conrelid = 'detections'::regclass 
AND conname = 'detections_patchcore_score_check';
