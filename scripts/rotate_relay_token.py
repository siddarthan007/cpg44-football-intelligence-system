"""Make a new relay token and put it in this laptop's .env.

The token has three homes, and all three must change together or the vest
stops reaching the relay:

1. the relay server's environment (CPG44_RELAY_TOKEN), then restart it;
2. this laptop's .env, which this script updates (the old file is kept as
   .env.before-rotation);
3. the vest firmware: flash it again from Flash a vest, which writes the
   new token into the ignored generated_config.h.

The new token is written to .env and to a file only you can read
(data/live/new_relay_token.txt, mode 600) for copying to the server. It is
never printed.

    python scripts/rotate_relay_token.py            # dry run: says what it would do
    python scripts/rotate_relay_token.py --apply    # writes the new token
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the new token (otherwise only explain)")
    args = ap.parse_args()
    env = ROOT / ".env"
    text = env.read_text(encoding="utf-8") if env.is_file() else ""
    if not re.search(r"^CPG44_RELAY_TOKEN=", text, flags=re.M):
        print(".env has no CPG44_RELAY_TOKEN line; nothing to rotate")
        return 1
    if not args.apply:
        print("Dry run. With --apply this replaces CPG44_RELAY_TOKEN in .env (backup kept) and saves the new\n"
              "token to data/live/new_relay_token.txt. Then: set it on the relay server and restart it, restart\n"
              "this app, and flash the vest again.")
        return 0
    token = secrets.token_urlsafe(48)
    shutil.copy2(env, env.with_name(".env.before-rotation"))
    env.write_text(re.sub(r"^CPG44_RELAY_TOKEN=.*$", f"CPG44_RELAY_TOKEN={token}", text, flags=re.M), encoding="utf-8")
    out = ROOT / "data" / "live" / "new_relay_token.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(token + "\n", encoding="utf-8")
    os.chmod(out, 0o600)
    print(f"New token written to .env and {out.relative_to(ROOT)} (not shown here).\n"
          "Next: 1) set it as CPG44_RELAY_TOKEN on the relay server and restart the relay,\n"
          "      2) restart this app, 3) flash the vest again from Flash a vest,\n"
          "      4) delete the token file once the server has it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
