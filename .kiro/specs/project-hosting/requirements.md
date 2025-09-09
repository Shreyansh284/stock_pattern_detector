# Requirements Document

## Introduction

Deploy the Stock Pattern Detection app (FastAPI backend + React frontend) to free hosting platforms.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to deploy the backend to Render for free, so that the API is accessible online.

#### Acceptance Criteria

1. WHEN deployed THEN the FastAPI app SHALL run on Render
2. WHEN API calls are made THEN they SHALL respond under 3 seconds
3. WHEN the app starts THEN it SHALL install dependencies from requirements.txt
4. WHEN CORS is configured THEN it SHALL accept requests from frontend domain

### Requirement 2

**User Story:** As a developer, I want to deploy the frontend to Vercel for free, so that users can access the web interface.

#### Acceptance Criteria

1. WHEN deployed THEN the React app SHALL be accessible via HTTPS
2. WHEN users navigate THEN routing SHALL work with proper fallbacks
3. WHEN built THEN it SHALL compile TypeScript and optimize assets
4. WHEN loaded THEN it SHALL work on mobile and desktop

### Requirement 3

**User Story:** As a developer, I want to configure production URLs and environment variables, so that frontend and backend connect properly.

#### Acceptance Criteria

1. WHEN frontend makes API calls THEN it SHALL use production backend URL
2. WHEN environment variables are set THEN both apps SHALL use production config
3. WHEN API is down THEN frontend SHALL show proper error messages
4. WHEN deployed THEN both apps SHALL be properly connected

### Requirement 4

**User Story:** As a developer, I want deployment configuration files, so that hosting is automated and reliable.

#### Acceptance Criteria

1. WHEN deployment files exist THEN they SHALL include build commands and settings
2. WHEN deployed THEN package.json scripts SHALL work correctly
3. WHEN built THEN all dependencies SHALL install properly
4. WHEN tested THEN all features SHALL work after deployment