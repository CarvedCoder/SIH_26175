# Project Agent Instructions: DepthWizard

Welcome to the **DepthWizard** repository. This document outlines project architecture, technical guidelines, development workflows, and design systems for AI agents and developers working on this codebase.

---

## 1. Project Overview

**DepthWizard** is a prototype application for single/monocular image depth estimation and Digital Surface Model (DSM) generation (SIH project).

### Architecture
- **Frontend (`/frontend`)**:
  - **Framework**: React 19 + Vite
  - **Styling**: Modern CSS / modular component styles
  - **Capabilities**: Image upload, job submission, real-time status polling, visualization of depth maps and DSM outputs.
- **Backend (`/backend`)**:
  - **Framework**: FastAPI + Uvicorn + Pydantic v2
  - **Routing structure**: `app/api/routes/` (`health`, `imagery`, `processing`)
  - **Services**: `app/services/` (ML processing pipeline integration for relative & metric depth calibration)
  - **Schemas**: `app/schemas/` (Pydantic models for job management and status tracking)

---

## 2. Integrated Skills & Design Philosophy

### Installed Skills
- **`frontend-design`** (located at `.agents/skills/frontend-design/SKILL.md`):
  - **Distinctive Visual Identity**: Approach UI as a design lead crafting a bespoke visual identity specific to satellite / depth imagery and spatial elevation tooling. Avoid generic AI aesthetics (e.g. cookie-cutter warm cream/serif or generic black/acid-green templates).
  - **Hero as a Thesis**: Feature real artifacts (interactive depth map visualizer, point clouds, split-screen contrast sliders, elevation cross-sections) rather than generic stats.
  - **Intentional Typography & Color**: Pair deliberate typefaces (technical data/mono utility paired with crisp sans-serif), use purposeful elevation/topography-inspired palette tokens.
  - **User-Centric Language**: Use action-oriented, domain-precise vocabulary ("Process Elevation Model", "Calibrate Metric Scale", "Export Point Cloud").

---

## 3. Development Workflow & Commands

### Backend
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend API server
uvicorn app.main:app --reload --port 8000
```
- API Docs available at `http://localhost:8000/docs`

### Frontend
```bash
cd frontend
npm install
npm run dev
```
- Local dev server available at `http://localhost:5173`

---

## 4. API Specification

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Health check endpoint |
| `/api/imagery/upload` | `POST` | Upload source image for depth analysis |
| `/api/processing/{job_id}` | `POST` | Trigger depth/DSM processing job |
| `/api/processing/{job_id}/status` | `GET` | Poll job processing status (`uploaded`, `processing`, `completed`, `failed`) |
| `/api/processing/{job_id}/result` | `GET` | Retrieve processed depth map and DSM metadata |

---

## 5. Agent Guidelines
- When implementing UI features, consult `.agents/skills/frontend-design/SKILL.md`.
- Keep code clean, modular, and typed (Pydantic schemas for backend, PropTypes/TypeScript/JSDoc conventions for frontend).
- Maintain responsiveness, accessibility, and high performance during high-resolution raster/image processing.
