-- ============================================================================
-- SUPABASE DATABASE SCHEMA FOR INDUSTRIAL INSPECTION SYSTEM
-- Copy and paste this into Supabase SQL Editor
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- TABLE 1: scans
-- Stores scan session information
-- ============================================================================
CREATE TABLE IF NOT EXISTS scans (
    scan_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    side VARCHAR(10) CHECK (side IN ('front', 'back', 'left', 'right', 'all')),
    total_components INTEGER DEFAULT 0,
    defective_count INTEGER DEFAULT 0,
    normal_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    amr_position FLOAT,
    camera_mode VARCHAR(20),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX idx_scans_start_time ON scans(start_time DESC);
CREATE INDEX idx_scans_status ON scans(status);

-- ============================================================================
-- TABLE 2: frames
-- Stores individual camera frame information
-- ============================================================================
CREATE TABLE IF NOT EXISTS frames (
    frame_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id UUID NOT NULL REFERENCES scans(scan_id) ON DELETE CASCADE,
    camera_id VARCHAR(10) NOT NULL CHECK (camera_id IN ('L', 'C', 'R', 'left', 'center', 'right')),
    height_level FLOAT NOT NULL,
    side VARCHAR(10) NOT NULL,
    image_url TEXT,
    image_path TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    detection_count INTEGER DEFAULT 0,
    processing_time FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_frames_scan_id ON frames(scan_id);
CREATE INDEX idx_frames_camera_id ON frames(camera_id);
CREATE INDEX idx_frames_height_level ON frames(height_level);
CREATE INDEX idx_frames_timestamp ON frames(timestamp DESC);

-- ============================================================================
-- TABLE 3: detections
-- Stores individual component detections with anomaly scores
-- ============================================================================
CREATE TABLE IF NOT EXISTS detections (
    detection_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    frame_id UUID NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
    scan_id UUID NOT NULL REFERENCES scans(scan_id) ON DELETE CASCADE,
    component_name VARCHAR(100) NOT NULL,
    bbox JSONB NOT NULL, -- {x1, y1, x2, y2}
    yolo_conf FLOAT NOT NULL CHECK (yolo_conf >= 0 AND yolo_conf <= 1),
    patchcore_status VARCHAR(20) CHECK (patchcore_status IN ('NORMAL', 'DEFECTIVE', 'UNKNOWN', 'ERROR')),
    patchcore_score FLOAT CHECK (patchcore_score >= 0),
    heatmap_url TEXT,
    crop_url TEXT,
    confidence_level VARCHAR(20),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_detections_frame_id ON detections(frame_id);
CREATE INDEX idx_detections_scan_id ON detections(scan_id);
CREATE INDEX idx_detections_component_name ON detections(component_name);
CREATE INDEX idx_detections_status ON detections(patchcore_status);
CREATE INDEX idx_detections_created_at ON detections(created_at DESC);

-- ============================================================================
-- TABLE 4: component_stats
-- Aggregated statistics per component type
-- ============================================================================
CREATE TABLE IF NOT EXISTS component_stats (
    stat_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component_name VARCHAR(100) NOT NULL UNIQUE,
    total_detected INTEGER DEFAULT 0,
    total_defective INTEGER DEFAULT 0,
    total_normal INTEGER DEFAULT 0,
    avg_anomaly_score FLOAT,
    max_anomaly_score FLOAT,
    min_anomaly_score FLOAT,
    last_detected TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index
CREATE INDEX idx_component_stats_name ON component_stats(component_name);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Recent scans with summary
CREATE OR REPLACE VIEW recent_scans AS
SELECT 
    s.scan_id,
    s.start_time,
    s.end_time,
    s.side,
    s.status,
    s.total_components,
    s.defective_count,
    s.normal_count,
    COUNT(DISTINCT f.frame_id) as frame_count,
    EXTRACT(EPOCH FROM (COALESCE(s.end_time, NOW()) - s.start_time)) as duration_seconds
FROM scans s
LEFT JOIN frames f ON s.scan_id = f.scan_id
GROUP BY s.scan_id
ORDER BY s.start_time DESC;

-- View: Detection summary by component
CREATE OR REPLACE VIEW detection_summary AS
SELECT 
    component_name,
    COUNT(*) as total_count,
    SUM(CASE WHEN patchcore_status = 'DEFECTIVE' THEN 1 ELSE 0 END) as defective_count,
    SUM(CASE WHEN patchcore_status = 'NORMAL' THEN 1 ELSE 0 END) as normal_count,
    AVG(patchcore_score) as avg_anomaly_score,
    AVG(yolo_conf) as avg_yolo_confidence
FROM detections
GROUP BY component_name
ORDER BY total_count DESC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Update scan statistics
CREATE OR REPLACE FUNCTION update_scan_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE scans
    SET 
        total_components = (
            SELECT COUNT(*) FROM detections WHERE scan_id = NEW.scan_id
        ),
        defective_count = (
            SELECT COUNT(*) FROM detections 
            WHERE scan_id = NEW.scan_id AND patchcore_status = 'DEFECTIVE'
        ),
        normal_count = (
            SELECT COUNT(*) FROM detections 
            WHERE scan_id = NEW.scan_id AND patchcore_status = 'NORMAL'
        ),
        updated_at = NOW()
    WHERE scan_id = NEW.scan_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function: Update component statistics
CREATE OR REPLACE FUNCTION update_component_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO component_stats (
        component_name,
        total_detected,
        total_defective,
        total_normal,
        avg_anomaly_score,
        max_anomaly_score,
        min_anomaly_score,
        last_detected
    )
    SELECT 
        NEW.component_name,
        1,
        CASE WHEN NEW.patchcore_status = 'DEFECTIVE' THEN 1 ELSE 0 END,
        CASE WHEN NEW.patchcore_status = 'NORMAL' THEN 1 ELSE 0 END,
        NEW.patchcore_score,
        NEW.patchcore_score,
        NEW.patchcore_score,
        NOW()
    ON CONFLICT (component_name) DO UPDATE SET
        total_detected = component_stats.total_detected + 1,
        total_defective = component_stats.total_defective + 
            CASE WHEN NEW.patchcore_status = 'DEFECTIVE' THEN 1 ELSE 0 END,
        total_normal = component_stats.total_normal + 
            CASE WHEN NEW.patchcore_status = 'NORMAL' THEN 1 ELSE 0 END,
        avg_anomaly_score = (
            SELECT AVG(patchcore_score) 
            FROM detections 
            WHERE component_name = NEW.component_name
        ),
        max_anomaly_score = GREATEST(component_stats.max_anomaly_score, NEW.patchcore_score),
        min_anomaly_score = LEAST(component_stats.min_anomaly_score, NEW.patchcore_score),
        last_detected = NOW(),
        updated_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger: Update scan stats when detection is inserted
CREATE TRIGGER trigger_update_scan_stats
AFTER INSERT ON detections
FOR EACH ROW
EXECUTE FUNCTION update_scan_stats();

-- Trigger: Update component stats when detection is inserted
CREATE TRIGGER trigger_update_component_stats
AFTER INSERT ON detections
FOR EACH ROW
EXECUTE FUNCTION update_component_stats();

-- Trigger: Update frame detection count
CREATE OR REPLACE FUNCTION update_frame_detection_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE frames
    SET detection_count = (
        SELECT COUNT(*) FROM detections WHERE frame_id = NEW.frame_id
    )
    WHERE frame_id = NEW.frame_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_frame_detection_count
AFTER INSERT ON detections
FOR EACH ROW
EXECUTE FUNCTION update_frame_detection_count();

-- ============================================================================
-- ROW LEVEL SECURITY (Optional - enable if needed)
-- ============================================================================

-- Enable RLS
-- ALTER TABLE scans ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE frames ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE detections ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE component_stats ENABLE ROW LEVEL SECURITY;

-- Create policies (example - adjust as needed)
-- CREATE POLICY "Allow all operations for service role" ON scans FOR ALL USING (true);
-- CREATE POLICY "Allow all operations for service role" ON frames FOR ALL USING (true);
-- CREATE POLICY "Allow all operations for service role" ON detections FOR ALL USING (true);
-- CREATE POLICY "Allow all operations for service role" ON component_stats FOR ALL USING (true);

-- ============================================================================
-- STORAGE BUCKETS (Run these in Supabase Dashboard > Storage)
-- ============================================================================

-- Create storage bucket for scan images
-- INSERT INTO storage.buckets (id, name, public) VALUES ('scans', 'scans', true);

-- Create storage bucket for heatmaps
-- INSERT INTO storage.buckets (id, name, public) VALUES ('heatmaps', 'heatmaps', true);

-- Create storage bucket for crops
-- INSERT INTO storage.buckets (id, name, public) VALUES ('crops', 'crops', true);

-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================

-- Get latest scan with all detections
-- SELECT s.*, d.* 
-- FROM scans s
-- LEFT JOIN detections d ON s.scan_id = d.scan_id
-- WHERE s.scan_id = '<scan_id>'
-- ORDER BY d.created_at;

-- Get defective components from latest scan
-- SELECT * FROM detections 
-- WHERE scan_id = (SELECT scan_id FROM scans ORDER BY start_time DESC LIMIT 1)
-- AND patchcore_status = 'DEFECTIVE';

-- Get component statistics
-- SELECT * FROM component_stats ORDER BY total_detected DESC;

-- Get scan performance metrics
-- SELECT 
--     DATE(start_time) as scan_date,
--     COUNT(*) as total_scans,
--     AVG(total_components) as avg_components,
--     AVG(defective_count) as avg_defects
-- FROM scans
-- WHERE status = 'completed'
-- GROUP BY DATE(start_time)
-- ORDER BY scan_date DESC;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Additional composite indexes for common queries
CREATE INDEX idx_detections_scan_status ON detections(scan_id, patchcore_status);
CREATE INDEX idx_frames_scan_camera ON frames(scan_id, camera_id);
CREATE INDEX idx_detections_component_status ON detections(component_name, patchcore_status);

-- ============================================================================
-- COMPLETED
-- ============================================================================

-- Verify tables created
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('scans', 'frames', 'detections', 'component_stats');
