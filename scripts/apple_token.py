"""Generate Apple MapKit JS JWT tokens for satellite map authentication."""

import os
import time

import jwt
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "apple_mapkit.yaml")


def load_mapkit_token(config_path=None):
    """Load Apple Developer credentials and generate a long-lived JWT.

    Returns:
        JWT token string for embedding in HTML.

    Raises:
        FileNotFoundError: If config file or private key is missing.
        ValueError: If config is incomplete.
    """
    if config_path is None:
        config_path = _CONFIG_PATH

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing {config_path}. "
            "Copy config/apple_mapkit.yaml.example to config/apple_mapkit.yaml "
            "and fill in your Apple Developer credentials."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    team_id = config.get("team_id", "")
    key_id = config.get("key_id", "")
    key_path = config.get("private_key_path", "")

    if not team_id or not key_id or not key_path:
        raise ValueError("apple_mapkit.yaml must contain team_id, key_id, and private_key_path")

    # Resolve relative path from repo root
    if not os.path.isabs(key_path):
        key_path = os.path.join(_REPO_ROOT, key_path)

    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Private key not found: {key_path}")

    with open(key_path) as f:
        private_key = f.read()

    now = int(time.time())
    payload = {
        "iss": team_id,
        "iat": now,
        "exp": now + 365 * 24 * 3600,  # 1 year
    }
    headers = {
        "kid": key_id,
        "typ": "JWT",
    }

    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)
