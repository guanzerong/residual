#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time

from resfit.rl_finetuning.utils.groot_adapter import _BaseInferenceClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client = _BaseInferenceClient(host=args.host, port=args.port, timeout_ms=3000)
            try:
                metadata = client.call_endpoint("get_metadata", requires_input=False)
            finally:
                client.close()
            payload = json.dumps(metadata, indent=2, sort_keys=True)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as handle:
                    handle.write(payload + "\n")
            print(payload)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2.0)
    raise SystemExit(f"GR00T server did not become ready: {last_error}")


if __name__ == "__main__":
    main()
