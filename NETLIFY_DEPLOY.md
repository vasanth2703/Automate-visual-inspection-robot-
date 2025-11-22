# Netlify Deployment Guide

## ⚠️ Important: Netlify is for Frontend Only!

Netlify hosts static files (HTML, CSS, JS). The Python backend needs a different platform.

## 🚀 Deploy Frontend to Netlify

### Method 1: Using netlify.toml (Automatic)

1. Push `netlify.toml` to GitHub
2. In Netlify Dashboard:
   - Connect to GitHub repository
   - Build settings will be auto-detected
   - Click "Deploy"

### Method 2: Manual Deploy

1. Go to Netlify Dashboard
2. Drag & drop the `frontend` folder
3. Done!

### Method 3: Netlify CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
cd frontend
netlify deploy --prod
```

## 🔧 Configure Frontend

Before deploying, update API URL in `frontend/react_dashboard.html`:

```javascript
// Change this line:
const API_BASE = 'http://localhost:8000';

// To your backend URL:
const API_BASE = 'https://your-backend-url.railway.app';
```

## 🖥️ Deploy Backend Separately

### Recommended Platforms for Python Backend:

#### 1. Railway (Easiest)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway up
```

**Or use Railway Dashboard:**
1. Go to railway.app
2. New Project → Deploy from GitHub
3. Select your repository
4. Add environment variables (SUPABASE_URL, SUPABASE_KEY)
5. Deploy!

#### 2. Render
1. Go to render.com
2. New → Web Service
3. Connect GitHub repository
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables
7. Deploy!

#### 3. Heroku
```bash
# Install Heroku CLI
# Create Procfile
echo "web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 📋 Complete Deployment Checklist

### Step 1: Deploy Backend
- [ ] Choose platform (Railway/Render/Heroku)
- [ ] Deploy Python backend
- [ ] Add environment variables
- [ ] Get backend URL (e.g., https://your-app.railway.app)
- [ ] Test: https://your-backend-url/docs

### Step 2: Update Frontend
- [ ] Open `frontend/react_dashboard.html`
- [ ] Update `API_BASE` to backend URL
- [ ] Update Supabase credentials (if needed)
- [ ] Commit changes

### Step 3: Deploy Frontend
- [ ] Deploy to Netlify
- [ ] Get frontend URL (e.g., https://your-app.netlify.app)
- [ ] Test dashboard

### Step 4: Configure CORS
In `backend/main.py`, update CORS origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-app.netlify.app",  # Your Netlify URL
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🎯 Quick Deploy Commands

### Deploy Backend to Railway:
```bash
railway login
railway init
railway up
railway open
```

### Deploy Frontend to Netlify:
```bash
cd frontend
netlify deploy --prod
```

## ✅ Verify Deployment

1. **Backend**: Visit `https://your-backend-url/docs`
   - Should show FastAPI documentation
   
2. **Frontend**: Visit `https://your-frontend-url`
   - Should show dashboard
   - Check browser console (F12) for errors
   
3. **Test Integration**:
   - Click "Start New Scan" in dashboard
   - Check if it connects to backend
   - Verify data appears

## 🐛 Troubleshooting

### Netlify Build Fails
- **Cause**: Trying to build Python backend
- **Fix**: Use `netlify.toml` to deploy only frontend folder

### CORS Errors
- **Cause**: Backend not allowing frontend domain
- **Fix**: Update CORS origins in `backend/main.py`

### API Not Connecting
- **Cause**: Wrong API_BASE URL
- **Fix**: Update URL in `react_dashboard.html`

### Environment Variables Missing
- **Cause**: Not set in deployment platform
- **Fix**: Add SUPABASE_URL and SUPABASE_KEY in platform settings

## 📞 Support

- Railway: https://railway.app/help
- Render: https://render.com/docs
- Netlify: https://docs.netlify.com
