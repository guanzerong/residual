from __future__ import annotations

import asyncio
import dataclasses
import http
import logging
from pathlib import Path
import socket
import sys
import time
import traceback

import tyro
import websockets
import websockets.asyncio.server as _server
import websockets.frames


_TASK_PROMPTS = {
    "lift": "pick up the object on the table and hold it",
    "can": "pick up the coke can and place it on the correct place",
    "square": "pick a square nut and place it on a rod",
    "toolhang": "assemble a frame consisting of a base piece and hook piece by inserting the hook into the base, and hang a wrench on the hook",
}

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    openpi_root: str = "/data_all/gzr1/openpi"
    config: str = "pi05_robomimic_lcsth"
    checkpoint_dir: str = "/data_all/gzr1/openpi/checkpoints/pi05_robomimic_lcsth/pi05_robomimic_lcsth_lora_20260427_1625/10000"
    task: str = "square"
    default_prompt: str | None = None
    port: int = 8765
    host: str = "0.0.0.0"


class FeatureWebsocketPolicyServer:
    def __init__(self, policy, *, metadata: dict, host: str, port: int) -> None:
        self._policy = policy
        self._metadata = metadata
        self._host = host
        self._port = port

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        from openpi_client import msgpack_numpy

        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                payload = msgpack_numpy.unpackb(await websocket.recv())
                if isinstance(payload, dict) and "op" in payload and "obs" in payload:
                    op = str(payload["op"])
                    obs = payload["obs"]
                else:
                    op = "infer"
                    obs = payload

                infer_start = time.monotonic()
                if op == "infer":
                    result = self._policy.infer(obs)
                elif op == "infer_and_encode":
                    result = self._policy.infer_and_encode(obs)
                elif op == "infer_and_encode_batch":
                    result = self._policy.infer_and_encode_batch(obs)
                else:
                    raise ValueError(f"Unsupported operation: {op}")
                infer_time = time.monotonic() - infer_start

                result["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main(args: Args) -> None:
    openpi_root = Path(args.openpi_root).expanduser().resolve()
    sys.path.insert(0, str(openpi_root / "src"))
    sys.path.insert(0, str(openpi_root / "packages" / "openpi-client" / "src"))

    from openpi.policies import policy_config as openpi_policy_config
    from openpi.training import config as openpi_training_config

    prompt = args.default_prompt or _TASK_PROMPTS.get(args.task.lower())
    if prompt is None:
        raise ValueError(f"No default prompt available for task={args.task!r}.")

    train_config = openpi_training_config.get_config(args.config)
    policy = openpi_policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=prompt,
    )
    metadata = dict(policy.metadata)
    metadata["action_horizon"] = int(train_config.model.action_horizon)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Starting OpenPI feature server on %s:%s (host=%s ip=%s)", args.host, args.port, hostname, local_ip)
    server = FeatureWebsocketPolicyServer(policy, metadata=metadata, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
