#!/bin/bash
# Saves the LINE channel secret and access token into .env without showing them on screen.
cd "$(dirname "$0")/.." || exit 1

save() {  # save NAME VALUE
  local name="$1" value
  value="$(printf '%s' "$2" | tr -d '[:space:]')"
  if [ -z "$value" ]; then echo "  (empty, skipped $name)"; return; fi
  if grep -q "^$name=" .env; then
    sed -i '' "s|^$name=.*|$name=$value|" .env
  else
    printf '%s=%s\n' "$name" "$value" >> .env
  fi
  echo "  saved $name (${#value} characters)"
}

echo "Step A: paste the Channel secret, then press Enter (nothing will show):"
read -rs SECRET; echo
save LINE_CHANNEL_SECRET "$SECRET"; unset SECRET

echo "Step B: paste the Channel access token (long-lived), then press Enter (nothing will show):"
read -rs TOKEN; echo
save LINE_CHANNEL_ACCESS_TOKEN "$TOKEN"; unset TOKEN

echo
echo "DONE. You can go back to the chat."
