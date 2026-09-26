# DepthWizard frontend image: Vite build served via `vite preview`.
#
#   docker build -t depthwizard-frontend ./frontend
#
# VITE_* variables are baked into the bundle at BUILD time (Vite inlines
# import.meta.env). The default API base (http://localhost:8000/api/v1,
# from src/api/client.js) already matches the compose-published backend
# port, so no build args are needed for the standard local setup.

FROM node:22-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . ./
RUN npm run build

FROM node:22-slim

WORKDIR /app
RUN npm install -g vite@8 --no-audit --no-fund

COPY --from=build /app/dist ./dist

EXPOSE 5173

# vite preview serves the production dist/ build (same server `npm run
# preview` uses in local dev).
CMD ["vite", "preview", "--host", "0.0.0.0", "--port", "5173", "--strictPort"]
