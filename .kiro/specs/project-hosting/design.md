# Design Document

## Overview

Deploy the Stock Pattern Detection application using free hosting services:
- **Backend**: Render for FastAPI
- **Frontend**: Vercel for React app

## Architecture

```
User Browser
     ↓
Frontend (Vercel)
     ↓ API calls
Backend (Render)
     ↓ Data fetching
Yahoo Finance API
```

## Components and Interfaces

### Backend Deployment (Render)

**Render Setup:**
- Domain: `your-app-name.onrender.com`
- Free tier: 750 hours/month, sleeps after 15min inactivity
- Cold start: ~30 seconds wake-up time
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

**Required Files:**
- `requirements.txt` (already exists)
- Render dashboard config
- Production environment variables

### Frontend Deployment (Vercel)

**Vercel Setup:**
```bash
# Install Vercel CLI
npm install -g vercel
# Deploy
vercel --prod
```
- Domain: `your-app-name.vercel.app`
- Free tier: 100GB bandwidth, unlimited static sites
- Build command: `npm run build`
- Output directory: `dist`



**Required package.json Scripts:**
```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "deploy": "npm run build && vercel --prod"
  }
}
```

## Data Models

### Environment Variables

**Backend Environment Variables:**
```bash
# Render Dashboard
PORT=8000
CORS_ORIGINS=https://stock-pattern-detector.vercel.app
STOCK_DATA_DIR=/tmp/stock_data
PYTHONPATH=/app
```

**Frontend Environment Variables:**
```bash
# Vercel Dashboard
VITE_API_URL=https://stock-pattern-detector.onrender.com
```

**CORS Configuration Example:**
```python
# In backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://stock-pattern-detector.vercel.app",
        "http://localhost:5173"  # for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Configuration Files

**Backend - Render Configuration:**
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

**Frontend - vercel.json:**
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ],
  "buildCommand": "npm run build",
  "outputDirectory": "dist"
}
```

## Error Handling

### Backend Error Scenarios
- **API Rate Limits**: Implement caching and request throttling
- **Memory Limits**: Optimize pattern detection algorithms
- **Cold Starts**: Accept initial slower response times

### Frontend Error Scenarios
- **API Unavailable**: Show user-friendly error messages
- **Slow Responses**: Add loading states and timeouts
- **Network Issues**: Implement retry logic with exponential backoff

### Fallback Strategies
- **Backend Down**: Display cached data or maintenance message
- **Build Failures**: Keep previous deployment active
- **Environment Issues**: Provide clear error messages

## Testing Strategy

### Build Troubleshooting

**Common Backend Issues:**
- **Memory Limit**: Reduce batch sizes, optimize imports
- **Timeout**: Implement request caching, reduce API calls
- **Dependencies**: Pin versions in requirements.txt
- **Port Binding**: Use `--host 0.0.0.0 --port $PORT`

**Common Frontend Issues:**
- **Build Timeout**: Increase build timeout in platform settings
- **Environment Variables**: Prefix with `VITE_` for Vite access
- **Routing**: Add proper redirects for SPA
- **Asset Size**: Optimize images and bundle size

### Monitoring and Uptime

**Free Monitoring Tools:**
- **UptimeRobot**: 50 monitors free, 5-minute intervals
- **Pingdom**: 1 monitor free, 1-minute intervals
- **StatusCake**: 10 monitors free, 5-minute intervals

**Setup Example:**
```bash
# UptimeRobot monitors
Frontend: https://stock-pattern-detector.vercel.app
Backend Health: https://stock-pattern-detector.railway.app/stocks
```

**Cost Monitoring:**
- Railway: Check usage in dashboard, set spending limits
- Render: Monitor build minutes and bandwidth
- Vercel/Netlify: Track bandwidth and function invocations

### Rollback Procedures

**Render Rollback:**
- Use dashboard to redeploy previous commit
- Or push previous commit to trigger new build

**Vercel Rollback:**
- Use dashboard to promote previous deployment
- Or redeploy from previous Git commit

### Data Persistence Strategy

**Free Hosting Limitations:**
- No persistent file storage on Render
- Use `/tmp` for temporary files only
- Consider external storage for large datasets

**Solutions:**
- **Cache**: Use Redis Cloud free tier (30MB)
- **Database**: Use PlanetScale/Supabase free tier if needed
- **Files**: Use Cloudinary/AWS S3 free tier for static assets
- **Session**: Use memory-based storage (resets on restart)

## Deployment Process

### Initial Setup
1. **Repository**: Ensure code is in GitHub repository
2. **Backend**: Connect Render to repository
3. **Frontend**: Connect Vercel to repository
4. **Environment**: Configure production environment variables
5. **Testing**: Verify deployment works end-to-end

### Continuous Deployment
1. **Git Push**: Automatic deployment on main branch push
2. **Build Logs**: Monitor deployment logs for issues
3. **Rollback**: Use platform rollback features if needed
4. **Updates**: Regular dependency updates and security patches