#!/usr/bin/env python
"""Send a Feishu/Lark message via lark-cli (bot identity, P2P).

Usage: python tools/notify_lark.py "message text"
Exit 0 on success; prints the API response tail on failure but never raises
(notification failure must not crash the calling pipeline).
"""
import subprocess
import sys

USER_OPEN_ID = "ou_5d83eac1e938b5aee1e40f65d5755da9"


def notify(text: str) -> bool:
    try:
        r = subprocess.run(
            ["lark-cli", "im", "+messages-send", "--as", "bot",
             "--user-id", USER_OPEN_ID, "--text", text],
            capture_output=True, text=True, timeout=30, shell=True if sys.platform == "win32" else False)
        ok = r.returncode == 0 and '"ok": true' in (r.stdout or "")
        if not ok:
            print("notify failed:", (r.stdout or "")[-300:], (r.stderr or "")[-300:])
        return ok
    except Exception as e:
        print("notify exception:", e)
        return False


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "test"
    sys.exit(0 if notify(msg) else 1)
