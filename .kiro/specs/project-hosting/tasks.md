# Implementation Plan

- [x] 1. Create backend deployment configuration files (10 mins)







  - Update `backend/main.py` CORS origins for Vercel production domain
  - Create `.env.example` with required environment variables for Render
  - Document Render build and start commands in README
  - **Test**: `uvicorn backend.main:app --host 0.0.0.0 --port 8000` works locally
  - **Rollback**: Revert CORS changes if local testing fails
  - _Requirements: 1.1, 1.4, 4.1_

- [x] 2. Create frontend deployment configuration files (10 mins)





  - Create `vercel.json` in `frontend/` directory with SPA routing
  - Update `frontend/package.json` with deployment scripts
  - **Test**: `npm run build` completes without errors in `frontend/`
  - **Rollback**: Remove config files if build fails
  - _Requirements: 2.2, 2.3, 4.1_

- [x] 3. Update frontend API configuration (10 mins)




  - Create `frontend/.env.example` with `VITE_API_URL` variable for Render backend
  - Update API calls in `frontend/src/` to use `import.meta.env.VITE_API_URL`
  - Add fallback to localhost for development
  - **Test**: Frontend connects to local backend with environment variable
  - **Rollback**: Revert to hardcoded localhost URLs
  - _Requirements: 3.1, 3.2, 4.3_

- [-] 4. Set up Render account and deploy backend (20 mins)



  - Sign up at render.com with GitHub account
  - Create new web service and connect to repository
  - Set build command: `pip install -r requirements.txt`
  - Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
  - Set environment variables: `CORS_ORIGINS=https://your-app.vercel.app`
  - Deploy and get production URL (e.g., `your-app.onrender.com`)
  - **Test**: `curl https://your-app.onrender.com/stocks` returns stock list
  - **Rollback**: Delete Render service if deployment fails
  - _Requirements: 1.1, 1.2, 3.4_

- [ ] 5. Set up Vercel account and deploy frontend (15 mins)
  - Sign up at vercel.com with GitHub account
  - Import project from repository, select `frontend/` directory
  - Set environment variable: `VITE_API_URL=https://your-app.onrender.com`
  - Deploy and get production URL (e.g., `your-app.vercel.app`)
  - **Test**: Visit frontend URL and verify homepage loads
  - **Rollback**: Delete Vercel project if deployment fails
  - _Requirements: 2.1, 2.2, 3.1_

- [ ] 6. Update CORS configuration with actual domains (5 mins)
  - Update Render environment variable with actual Vercel domain
  - Redeploy backend with updated CORS settings
  - **Test**: Frontend can make API calls to backend without CORS errors
  - **Rollback**: Revert to wildcard CORS if specific domains fail
  - _Requirements: 1.4, 3.1, 3.2_

- [ ] 7. Test all application features end-to-end (20 mins)
  - Test homepage ticker tape functionality
  - Test pattern detection with sample stock (e.g., AAPL)
  - Test dashboard and navigation between pages
  - Test on mobile device and different browsers
  - **Test**: All features work without console errors
  - **Rollback**: Fix issues or revert to previous deployment
  - _Requirements: 3.3, 4.4, 2.4_

- [ ] 8. Set up monitoring and error handling (10 mins)
  - Sign up for UptimeRobot free account
  - Add monitors for frontend URL and backend `/stocks` endpoint
  - Test error scenarios (backend down, slow responses)
  - **Test**: Monitoring alerts work and error messages display properly
  - **Rollback**: Remove monitors if they cause false alerts
  - _Requirements: 3.3, 1.2, 4.4_

**Total Estimated Time**: 1.5-2 hours

**Dependencies**: 
- Task 1 must complete before Task 4
- Task 2-3 must complete before Task 5  
- Task 4-5 must complete before Task 6
- Task 6 must complete before Task 7

**Common Troubleshooting**:
- **Build Timeout**: Increase timeout in platform settings
- **CORS Errors**: Check environment variables and redeploy
- **404 Errors**: Verify SPA routing configuration
- **API Errors**: Check backend logs in Render dashboard