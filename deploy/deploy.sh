#!/usr/bin/env bash
#
# Deploys the Document Analyze API on the Orange Pi. Run by the self-hosted runner from
# .github/workflows/deploy.yml, but it can also be run by hand:
#
#   bash deploy/deploy.sh sha-abc1234
#
# Steps: check .env → copy compose → pull → new container → healthcheck.
# If the healthcheck fails it goes back to the previous version and exits with an error.

set -Eeuo pipefail

NEW_TAG="${1:-}"
APP_DIR="${APP_DIR:-/opt/document-analyzer}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8090/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"   # first start embeds the catalog → give it time
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$APP_DIR/.env"
SERVICE="api"

log()  { printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m✖ %s\033[0m\n' "$*" >&2; exit 1; }

[[ "$NEW_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || fail "Usage: deploy.sh <image_tag>  (got: '${NEW_TAG}')"
[[ -d "$APP_DIR" ]]  || fail "$APP_DIR does not exist (see the deployment notes, step 5)."
[[ -f "$ENV_FILE" ]] || fail "$ENV_FILE does not exist (see the deployment notes, step 5)."

# Required secrets must be present (values are never printed)
for var in OPENAI_API_KEY API_KEY; do
  grep -qE "^${var}=.+" "$ENV_FILE" || fail "$var is missing or empty in $ENV_FILE."
done

compose() {
  docker compose --project-directory "$APP_DIR" -f "$APP_DIR/docker-compose.prod.yml" "$@"
}

current_tag() { grep -E '^IMAGE_TAG=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true; }

set_tag() {
  if grep -qE '^IMAGE_TAG=' "$ENV_FILE"; then
    sed -i "s/^IMAGE_TAG=.*/IMAGE_TAG=$1/" "$ENV_FILE"
  else
    printf '\nIMAGE_TAG=%s\n' "$1" >> "$ENV_FILE"
  fi
}

wait_healthy() {
  local deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -fsS --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; do
    (( SECONDS >= deadline )) && return 1
    sleep 3
  done
}

# ── 1. Compose ──────────────────────────────────────────────────────────────────
log "Copying compose to $APP_DIR"
install -m 644 "$SCRIPT_DIR/docker-compose.prod.yml" "$APP_DIR/docker-compose.prod.yml"

PREV_TAG="$(current_tag)"
log "Current version: ${PREV_TAG:-<none>}  →  new: $NEW_TAG"

# ── 2. Pull ─────────────────────────────────────────────────────────────────────
# IMAGE_TAG from the environment wins over .env: .env keeps pointing to the previous
# version until step 3, so a failure here leaves nothing half-done.
export IMAGE_TAG="$NEW_TAG"
log "Pulling image"
compose pull "$SERVICE" || fail "Could not pull image $NEW_TAG. Nothing changed."

# ── 3. New container + healthcheck ──────────────────────────────────────────────
log "Starting $SERVICE $NEW_TAG"
set_tag "$NEW_TAG"
compose up -d "$SERVICE"

log "Waiting for $HEALTH_URL (max ${HEALTH_TIMEOUT}s)"
if wait_healthy; then
  printf '%s\n' "$NEW_TAG" > "$APP_DIR/.last_successful_tag"
  printf '%s %s\n' "$(date -Is)" "$NEW_TAG" >> "$APP_DIR/deploy-history.log"
  log "Pruning old images (>10 days and unused)"
  docker image prune -af --filter "until=240h" > /dev/null || true
  log "✅ Deployed $NEW_TAG"
  exit 0
fi

# ── 4. Automatic rollback ───────────────────────────────────────────────────────
printf '\n\033[1;31m✖ Healthcheck failed. Last logs:\033[0m\n'
compose logs --tail 80 "$SERVICE" || true

if [[ -n "$PREV_TAG" && "$PREV_TAG" != "$NEW_TAG" ]]; then
  log "Rolling back to $PREV_TAG"
  set_tag "$PREV_TAG"
  export IMAGE_TAG="$PREV_TAG"
  compose up -d "$SERVICE"
  wait_healthy && log "Rollback OK: the API is back on $PREV_TAG" \
               || printf '\033[1;31mThe rollback does not pass the healthcheck either: check by hand.\033[0m\n'
fi
fail "Deploy of $NEW_TAG failed."
