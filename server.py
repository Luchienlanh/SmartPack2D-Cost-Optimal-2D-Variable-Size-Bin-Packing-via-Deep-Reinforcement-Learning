from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch

from agents.a2c_agent import A2CNetwork
from agents.pg_agent import PolicyNetwork
from agents.ppo_agent import PPOAgent
from env.packing_env import Packing
from utils.device import device
from utils.heuristics import run_ffd_heuristic


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
MODEL_PATHS = {
    "ppo": ROOT_DIR / "ppo_model.pth",
    "a2c": ROOT_DIR / "a2c_train.pth",
    "pg": ROOT_DIR / "train_plc.pth",
}

RL_MAX_WIDTH = 300
RL_MAX_HEIGHT = 300
RL_MAX_ITEMS = 200
RL_MAX_BIN_TYPES = 5

_rl_agents: dict[str, torch.nn.Module] = {}


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def to_number(value, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{field_name} must be a number.") from exc
    if not torch.isfinite(torch.tensor(number)):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{field_name} must be finite.")
    return number


def parse_problem(payload: dict) -> tuple[list[dict], list[tuple[int, int]], list[dict], bool, str]:
    raw_bins = payload.get("bins")
    raw_pieces = payload.get("pieces")
    if not isinstance(raw_bins, list) or not raw_bins:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Add at least one bin type.")
    if not isinstance(raw_pieces, list) or not raw_pieces:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Add at least one piece.")

    bins: list[dict] = []
    for index, raw in enumerate(raw_bins):
        width = int(round(to_number(raw.get("width"), f"bins[{index}].width")))
        height = int(round(to_number(raw.get("height"), f"bins[{index}].height")))
        cost = to_number(raw.get("cost", 0), f"bins[{index}].cost")
        if width <= 0 or height <= 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Bin width and height must be positive.")
        if cost < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Bin cost cannot be negative.")
        name = str(raw.get("name") or f"B{index + 1}").strip() or f"B{index + 1}"
        bins.append({"index": index, "name": name, "width": width, "height": height, "cost": cost})

    items: list[tuple[int, int]] = []
    item_meta: list[dict] = []
    for group_index, raw in enumerate(raw_pieces):
        width = int(round(to_number(raw.get("width"), f"pieces[{group_index}].width")))
        height = int(round(to_number(raw.get("height"), f"pieces[{group_index}].height")))
        qty = int(round(to_number(raw.get("qty", 1), f"pieces[{group_index}].qty")))
        if width <= 0 or height <= 0 or qty <= 0:
            continue
        group = str(raw.get("group") or raw.get("name") or f"P{group_index + 1}").strip() or f"P{group_index + 1}"

        # The frontend usually sends expanded pieces. If qty is present on a grouped row,
        # expand it here so direct API callers can use either shape.
        copies = qty if "copy" not in raw and "id" not in raw else 1
        for copy in range(1, copies + 1):
            piece_id = str(raw.get("id") or f"{group}-{copy}").strip() or f"{group}-{copy}"
            item_meta.append(
                {
                    "id": piece_id,
                    "group": group,
                    "groupIndex": int(raw.get("groupIndex", group_index) or 0),
                    "copy": int(raw.get("copy", copy) or copy),
                    "width": width,
                    "height": height,
                }
            )
            # Packing stores item dimensions as (height, width).
            items.append((height, width))

    if not items:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Add at least one valid piece.")

    allow_rotation = bool(payload.get("allowRotation", True))
    solver = str(payload.get("solver") or payload.get("searchLevel") or "ffd").lower()
    return bins, items, item_meta, allow_rotation, solver


def normalize_rl_solver(solver: str) -> str:
    if solver in {"policy_gradient", "policy-gradient"}:
        return "pg"
    return solver


def load_rl_agent(solver: str) -> torch.nn.Module:
    solver = normalize_rl_solver(solver)
    if solver in _rl_agents:
        return _rl_agents[solver]

    checkpoint_path = MODEL_PATHS[solver]
    if not checkpoint_path.exists():
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{checkpoint_path.name} was not found.")

    action_size = (RL_MAX_ITEMS * 2) + RL_MAX_BIN_TYPES
    if solver == "ppo":
        agent = PPOAgent(
            height=RL_MAX_HEIGHT,
            width=RL_MAX_WIDTH,
            action_size=action_size,
            num_items=RL_MAX_ITEMS,
        ).to(device)
    elif solver == "a2c":
        agent = A2CNetwork(
            height=RL_MAX_HEIGHT,
            width=RL_MAX_WIDTH,
            action_size=action_size,
            num_items=RL_MAX_ITEMS,
        ).to(device)
    else:
        agent = PolicyNetwork(
            height=RL_MAX_HEIGHT,
            width=RL_MAX_WIDTH,
            action_size=action_size,
            num_items=RL_MAX_ITEMS,
        ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    for key, module_name in [
        ("frame_net_state_dict", "frame_net"),
        ("item_net_state_dict", "item_net"),
        ("actor_state_dict", "actor"),
        ("actor_net_state_dict", "actor"),
        ("critic_state_dict", "critic"),
        ("critic_net_state_dict", "critic"),
        ("policy_state_dict", "policy_head"),
    ]:
        module = getattr(agent, module_name, None)
        if key in checkpoint and module is not None:
            module.load_state_dict(checkpoint[key])

    agent.eval()
    _rl_agents[solver] = agent
    return agent


def run_rl_forward(agent: torch.nn.Module, solver: str, frame_4d: torch.Tensor, remain_2d: torch.Tensor) -> torch.Tensor:
    output = agent(frame_4d, remain_2d)
    if normalize_rl_solver(solver) in {"ppo", "a2c"}:
        return output[0]
    return output


def bin_can_fit_remaining(bin_cfg: dict, remain_items: list[list[int]], num_items: int, allow_rotation: bool) -> bool:
    for item_index in range(num_items):
        h, w = remain_items[item_index]
        if h == 0 and w == 0:
            continue
        if h <= bin_cfg["height"] and w <= bin_cfg["width"]:
            return True
        if allow_rotation and w <= bin_cfg["height"] and h <= bin_cfg["width"]:
            return True
    return False


def constrain_rl_valid_actions(
    env: Packing,
    action_space: list[tuple],
    valid_idx: list[int],
    bins: list[dict],
    allow_rotation: bool,
) -> list[int]:
    if not allow_rotation:
        valid_idx = [idx for idx in valid_idx if action_space[idx][0] == "open" or not action_space[idx][1]]

    placement_idx = [idx for idx in valid_idx if action_space[idx][0] != "open"]
    if placement_idx:
        return placement_idx

    open_idx = [idx for idx in valid_idx if action_space[idx][0] == "open"]
    return [
        idx
        for idx in open_idx
        if bin_can_fit_remaining(bins[action_space[idx][1]], env.remain_items, env.num_items, allow_rotation)
    ]


def solve_with_ffd(bins: list[dict], items: list[tuple[int, int]], item_meta: list[dict], allow_rotation: bool) -> dict:
    max_width = max(bin_cfg["width"] for bin_cfg in bins)
    max_height = max(bin_cfg["height"] for bin_cfg in bins)
    result = run_ffd_heuristic(
        bins,
        items,
        max_width=max_width,
        max_height=max_height,
        max_items=max(len(items), 1),
        allow_rotation=allow_rotation,
    )
    return serialize_result(
        placed_items=result["placed_items"],
        opened_bins=result["opened_bins"],
        item_meta=item_meta,
        total_cost=result["cost"],
        objective="Core Packing environment with FFD ordering",
        variant="backend/ffd",
        notes=[],
    )


def solve_with_rl(solver: str, bins: list[dict], items: list[tuple[int, int]], item_meta: list[dict], allow_rotation: bool) -> dict:
    solver = normalize_rl_solver(solver)
    if len(items) > RL_MAX_ITEMS:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{solver.upper()} supports up to {RL_MAX_ITEMS} pieces.")
    if len(bins) > RL_MAX_BIN_TYPES:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{solver.upper()} supports up to {RL_MAX_BIN_TYPES} bin types.")
    if max(bin_cfg["width"] for bin_cfg in bins) > RL_MAX_WIDTH or max(bin_cfg["height"] for bin_cfg in bins) > RL_MAX_HEIGHT:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{solver.upper()} checkpoint expects bins no larger than 300 x 300.")

    agent = load_rl_agent(solver)
    env = Packing(
        bin_types=bins,
        items_or_height=items,
        max_width=RL_MAX_WIDTH,
        max_height=RL_MAX_HEIGHT,
        max_items=RL_MAX_ITEMS,
    )
    action_space = [(i, rot) for i in range(RL_MAX_ITEMS) for rot in [False, True]]
    action_space.extend(("open", b_idx) for b_idx in range(len(bins)))

    max_open_bins = max(1, len(items)) + 1
    max_steps = max(80, len(items) * 3 + len(bins) * 2)
    notes: list[str] = []

    for _ in range(max_steps):
        if env.is_done():
            break

        valid_idx = constrain_rl_valid_actions(
            env,
            action_space,
            env.get_valid_actions(action_space, allow_rotation=allow_rotation),
            bins,
            allow_rotation,
        )
        if env.opened_bins_count >= max_open_bins:
            valid_idx = [idx for idx in valid_idx if action_space[idx][0] != "open"]
        if not valid_idx:
            break

        frame, remain = env.get_state()
        frame_4d = frame.unsqueeze(0).unsqueeze(0).float().to(device)
        remain_2d = remain.view(1, -1).float().to(device)
        with torch.no_grad():
            logits = run_rl_forward(agent, solver, frame_4d, remain_2d)
        logits = logits.squeeze(0).float()

        mask = torch.full_like(logits, -1e9)
        mask[valid_idx] = 0.0
        action_index = int(torch.argmax(logits + mask).item())
        action = action_space[action_index]

        env.place(action)
    else:
        notes.append(f"{solver.upper()} stopped after the safety limit of {max_steps} actions.")

    return serialize_result(
        placed_items=env.placed_items,
        opened_bins=env.opened_bins,
        item_meta=item_meta,
        total_cost=env.total_bin_cost,
        objective=f"Core Packing environment with {solver.upper()} checkpoint",
        variant=f"backend/{solver}",
        notes=notes,
    )


def serialize_result(
    placed_items: list[tuple],
    opened_bins: list[dict],
    item_meta: list[dict],
    total_cost: float,
    objective: str,
    variant: str,
    notes: list[str],
) -> dict:
    rendered_bins = []
    type_counts: dict[int, int] = {}
    for bin_number, bin_cfg in enumerate(opened_bins, start=1):
        type_index = int(bin_cfg.get("index", 0))
        type_counts[type_index] = type_counts.get(type_index, 0) + 1
        type_name = str(bin_cfg.get("name") or f"B{type_index + 1}")
        rendered_bins.append(
            {
                "id": bin_number,
                "name": f"{type_name}{type_counts[type_index]}",
                "typeName": type_name,
                "typeIndex": type_index,
                "width": int(bin_cfg["width"]),
                "height": int(bin_cfg["height"]),
                "cost": float(bin_cfg["cost"]),
                "placements": [],
            }
        )

    placements = []
    placed_indices = set()
    for step_index, item in enumerate(placed_items, start=1):
        item_index, row, col, item_h, item_w, rotated, bin_number = item[:7]
        if item_index >= len(item_meta) or bin_number < 1 or bin_number > len(rendered_bins):
            continue
        meta = item_meta[item_index]
        bin_info = rendered_bins[bin_number - 1]
        placement = {
            "step": step_index,
            "pieceId": meta["id"],
            "group": meta["group"],
            "groupIndex": meta["groupIndex"],
            "copy": meta["copy"],
            "sourceWidth": meta["width"],
            "sourceHeight": meta["height"],
            "x": int(col),
            "y": int(row),
            "width": int(item_w),
            "height": int(item_h),
            "rotated": bool(rotated),
            "binId": int(bin_number),
            "binName": bin_info["name"],
            "binType": bin_info["typeName"],
            "binTypeIndex": bin_info["typeIndex"],
        }
        bin_info["placements"].append(placement)
        placements.append(placement)
        placed_indices.add(item_index)

    unplaced = [
        {
            "id": meta["id"],
            "group": meta["group"],
            "groupIndex": meta["groupIndex"],
            "copy": meta["copy"],
            "width": meta["width"],
            "height": meta["height"],
        }
        for index, meta in enumerate(item_meta)
        if index not in placed_indices
    ]

    placed_area = sum(item["width"] * item["height"] for item in placements)
    total_area = sum(bin_info["width"] * bin_info["height"] for bin_info in rendered_bins)
    waste = total_area - placed_area
    return {
        "bins": rendered_bins,
        "steps": placements,
        "placements": placements,
        "unplaced": unplaced,
        "placedArea": placed_area,
        "totalArea": total_area,
        "totalCost": total_cost,
        "waste": waste,
        "utilization": (placed_area / total_area * 100) if total_area else 0,
        "objective": objective,
        "variant": variant,
        "variantCount": 1,
        "notes": notes,
    }


def result_rank(result: dict) -> tuple:
    return (
        len(result["unplaced"]),
        float(result["totalCost"]),
        len(result["bins"]),
        float(result["waste"]),
        -float(result["utilization"]),
    )


def choose_rotation_result(rotation_result: dict, no_rotation_result: dict) -> dict:
    if result_rank(no_rotation_result) < result_rank(rotation_result):
        selected = no_rotation_result
        selected["variant"] = f"{selected['variant']}/no-rotation-selected"
        selected["notes"] = [
            *selected.get("notes", []),
            "Rotation was allowed, but the non-rotating candidate had a better objective.",
        ]
        return selected

    rotation_result["notes"] = [
        *rotation_result.get("notes", []),
        "Rotation was allowed and the rotating action space was selected.",
    ]
    return rotation_result


def solve_payload(payload: dict) -> dict:
    bins, items, item_meta, allow_rotation, solver = parse_problem(payload)
    solver = normalize_rl_solver(solver)
    if solver in MODEL_PATHS:
        if allow_rotation:
            return choose_rotation_result(
                solve_with_rl(solver, bins, items, item_meta, True),
                solve_with_rl(solver, bins, items, item_meta, False),
            )
        return solve_with_rl(solver, bins, items, item_meta, allow_rotation)
    if solver in {"ffd", "fast", "balanced", "deep", "core"}:
        if allow_rotation:
            return choose_rotation_result(
                solve_with_ffd(bins, items, item_meta, True),
                solve_with_ffd(bins, items, item_meta, False),
            )
        return solve_with_ffd(bins, items, item_meta, allow_rotation)
    raise ApiError(HTTPStatus.BAD_REQUEST, f"Unknown solver '{solver}'.")


class PackingRequestHandler(BaseHTTPRequestHandler):
    server_version = "SmartPack2D/1.0"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.write_json({
                "ok": True,
                "device": str(device),
                "models": {name: path.exists() for name, path in MODEL_PATHS.items()},
            })
            return
        self.serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/pack":
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object.")
            result = solve_payload(payload)
            self.write_json(result)
        except ApiError as exc:
            self.write_json({"error": exc.message}, exc.status)
        except json.JSONDecodeError:
            self.write_json({"error": "Invalid JSON body."}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # Keep the UI responsive and return the backend reason.
            self.write_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_static(self, request_path: str):
        if request_path in {"", "/"}:
            file_path = FRONTEND_DIR / "index.html"
        else:
            relative = Path(unquote(request_path).lstrip("/"))
            file_path = (FRONTEND_DIR / relative).resolve()
            if FRONTEND_DIR.resolve() not in file_path.parents and file_path != FRONTEND_DIR.resolve():
                self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return

        if not file_path.exists() or not file_path.is_file():
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        with file_path.open("rb") as handle:
            self.wfile.write(handle.read())

    def write_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description="Run the SmartPack2D backend and frontend server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), PackingRequestHandler)
    print(f"SmartPack2D server running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
