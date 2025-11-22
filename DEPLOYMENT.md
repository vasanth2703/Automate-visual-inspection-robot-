# Deployment Guide

## 🚀 Deploy Industrial Inspection System

### Prerequisites

- Python 3.12
- Git
- Supabase account
- GPU (optional, for training)

### Quick Deploy

```bash
# 1. Clone repository
git clone https://github.com/vasanth2703/Automate-visual-inspection-robot-.git
cd Automate-visual-inspection-robot-

# 2. Install dependencies
py -3.12 -m pip install -r requirements.txt

# 3. Setup environment
# Create .env file with your Supabase credentials
# SUPABASE_URL=your_url
# SUPABASE_KEY=your_key

# 4. Setup database
# Copy supabase_schema.sql to Supabase SQL Editor and run

# 5. Setup storage (optional)
py -3.12 setup_supabase_storage.py

# 6. Test system
py -3.12 test_complete_integration.py

# 7. Start backend
py -3.12 run_server.py

# 8. Open dashboard
py -3.12 launch_dashboard.py
```

### Production Deployment

#### Backend (FastAPI)

**Option 1: Docker**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Option 2: Cloud Platform**
- Deploy to: Railway, Render, Heroku, AWS, GCP, Azure
- Set environment variables (SUPABASE_URL, SUPABASE_KEY)
- Run: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

#### Frontend (React Dashboard)

**Option 1: Static Hosting**
- Deploy `frontend/react_dashboard.html` to:
  - Netlify
  - Vercel
  - GitHub Pages
  - Cloudflare Pages

**Option 2: Update API URL**
```javascript
// In react_dashboard.html, update:
const API_BASE = 'https://your-backend-url.com';
```

### Environment Variables

Create `.env` file:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
```

### Database Setup

1. Go to Supabase Dashboard
2. SQL Editor → New Query
3. Copy content from `supabase_schema.sql`
4. Run query
5. Verify tables created

### Storage Setup

```bash
py -3.12 setup_supabase_storage.py
```

Or manually create buckets:
- `scans` (public)
- `heatmaps` (public)
- `crops` (public)

### Model Training

**Note:** Models are NOT included in git (too large)

```bash
# Train YOLO (requires dataset in images/ folder)
py -3.12 system_control.py --train --quick

# Or use pre-trained models
# Download from releases and place in:
# - runs/detect/component_detector/weights/best.pt
# - models/patchcore/*.pkl
```

### Testing

```bash
# Complete integration test
py -3.12 test_complete_integration.py

# Supabase connection test
py -3.12 test_supabase.py

# Simulation test
py -3.12 test_simulation.py
```

### Monitoring

- Backend: http://your-backend-url/docs
- Dashboard: http://your-frontend-url
- Supabase: https://app.supabase.com

### Troubleshooting

**Backend won't start:**
- Check Python version: `py -3.12 --version`
- Install dependencies: `py -3.12 -m pip install -r requirements.txt`
- Check port 8000 is available

**Database errors:**
- Verify .env credentials
- Check schema is created
- Run: `py -3.12 test_supabase.py`

**Dashboard not updating:**
- Check browser console (F12)
- Verify API_BASE URL
- Check Supabase real-time enabled

**No AI detections:**
- Train models first
- Or download pre-trained models
- Check: `py -3.12 system_control.py --status`

### Performance

**Recommended Specs:**
- CPU: 4+ cores
- RAM: 8GB+ (16GB for training)
- GPU: NVIDIA with CUDA (optional)
- Storage: 10GB+

**Optimization:**
- Use GPU for faster inference
- Enable caching
- Use CDN for frontend
- Database indexing (already configured)

### Security

**Production Checklist:**
- [ ] Use environment variables
- [ ] Enable HTTPS
- [ ] Set CORS properly
- [ ] Use service role key (not anon key)
- [ ] Enable RLS in Supabase
- [ ] Regular backups
- [ ] Monitor logs

### Support

- Issues: GitHub Issues
- Documentation: README.md
- Quick Fix: QUICK_FIX.md
- Supabase Setup: SUPABASE_SETUP.md

### License

See LICENSE file
