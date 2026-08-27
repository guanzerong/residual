from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groot-root", required=True)
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.5-3B")
    parser.add_argument("--task", default="square")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--token-target-count", type=int, default=32)
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--base-image-key", default="observation.images.agentview")
    parser.add_argument("--wrist-image-key", default="observation.images.robot0_eye_in_hand")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.groot_root).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(repo_root))

    from gr00t.eval.service import BaseInferenceServer

    from resfit.rl_finetuning.utils.groot_feature_policy import GROOTFeaturePolicy

    policy = GROOTFeaturePolicy(
        groot_root=str(root),
        model_path=args.model_path,
        task_name=args.task,
        default_prompt=args.default_prompt,
        token_target_count=int(args.token_target_count),
        base_image_key=args.base_image_key,
        wrist_image_key=args.wrist_image_key,
    )

    server = BaseInferenceServer(host=args.host, port=args.port)
    server.register_endpoint("get_action", policy.infer)
    server.register_endpoint("get_action_and_features", policy.infer_and_encode)
    server.register_endpoint("encode_observation_features", lambda obs: policy.infer_and_encode(obs))
    server.register_endpoint("get_metadata", lambda: dict(policy.metadata), requires_input=False)
    server.run()


if __name__ == "__main__":
    main()
