# Aperture is a fully client-side app (no backend, no
# GPU) — this image just builds the static bundle and serves it. Any
# static file host works too; this is the containerized equivalent.

# ---- build stage ----
FROM node:22-alpine AS build
WORKDIR /app

# Copying the whole workspace before `npm ci` (rather than just the root
# package.json) is deliberate: this is an npm-workspaces monorepo, so npm
# needs every package's package.json present to link the workspace
# correctly. That trades away some Docker layer caching (a source-only
# change also invalidates the npm ci layer) for a Dockerfile that keeps
# working as packages are added or removed, which happens often here.
COPY . .
RUN npm ci
RUN npm run build

# ---- runtime stage ----
FROM nginx:alpine AS runtime
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/apps/web/dist /usr/share/nginx/html
EXPOSE 80
