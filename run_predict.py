#!/usr/bin/env python3

import os
import sys
import json
import csv
from typing import List, Tuple
from io import BytesIO
import base64
import cgi
from datetime import datetime

# Reduce backend noise and avoid TF/Flax/TorchVision imports inside Transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# --------------------------------------------
FILE_EXTENSION = 'fractal20220817_data'  # 'austin_buds_dataset_converted_externally_to_rlds' #'fractal20220817'
# --------------------------------------------
SERVER_MODE = False
# --------------------------------------------
# Per-dataset gripper post-processing rules.
# Keyed by `unnorm_key` (dataset name used for un-normalization/stats).
# Each rule can:
#   - invert: flip semantics via (1 - g)
#   - binarize: snap to {0,1} using `threshold`
#   - threshold: cutoff for binarization (default 0.5)
DATASET_GRIPPER_RULES: dict[str, dict] = {
    # Open-X RT-1 dataset
    "fractal20220817_data": {"invert": True, "binarize": True, "threshold": 0.5},
    # Local Unity dataset (closed=1 in raw, standardized to 0/1)
    "my_unity_robot_dataset": {"invert": True, "binarize": True, "threshold": 0.5},
}
# --------------------------------------------
USER_PROMPT = "In: What action should the robot take to pick rxbar chocolate from bottom drawer and place on counter?\n Out:"
# --------------------------------------------
MODEL_NAME = 'openvla/openvla-7b'
# --------------------------------------------
DATASET_STATS_PATH = None
DATASET_STATS_PATH = None
# --------------------------------------------
UN_NORMALIZED_DATASET_KEY = "fractal20220817_data"

# --------------------------------------------
class RobotBrain:
    """Small helper to load OpenVLA and run predictions on GIF frames.

    The class encapsulates device/dtype selection, model/processor loading,
    input preparation, and per-frame prediction + CSV logging.
    """

    def __init__(
            self,
    ) -> None:
        """Configure defaults and select device/dtype.

        Parameters can be overridden to change prompt, model source, or I/O paths.
        The model itself is lazy-loaded in load_openvla()/run_on_gif().
        """
        self.prompt = USER_PROMPT
        print(f'Using prompt: "{self.prompt}"')
        self.model_id = MODEL_NAME
        self.local_model_dir = None
        self.hf_token = None
        self.unnorm_key = UN_NORMALIZED_DATASET_KEY
        self.gif_path = os.path.join("../Out", f"{FILE_EXTENSION}.gif")
        self.csv_path = os.path.join("../Out", f"{FILE_EXTENSION}_gif_actions.csv")
        self.device, self.torch_dtype = self.select_device_dtype()
        self.proc: AutoProcessor | None = None
        self.model: AutoModelForVision2Seq | None = None

    @staticmethod
    def info(msg: str) -> None:
        """Print a timestamped log message."""
        print(f"[{datetime.now().strftime('%H:%M:%S .%f')}] {msg}")

    @staticmethod
    def require_timm_compatible() -> None:
        """Ensure timm is installed within the supported version range."""
        try:
            import timm  # type: ignore
            from packaging.version import Version
        except Exception as e:
            print("[ERROR] Missing dependency 'timm' in required range (>=0.9.10,<1.0.0).\n"
                  "Install with: pip install 'timm>=0.9.10,<1.0.0'\n"
                  f"Details: {e}")
            sys.exit(1)
        v = Version(getattr(timm, "__version__", "0"))
        if not (Version("0.9.10") <= v < Version("1.0.0")):
            print(f"[ERROR] Incompatible timm version: {v}. Required: >=0.9.10 and <1.0.0\n"
                  "Fix with: pip install 'timm>=0.9.10,<1.0.0'")
            sys.exit(1)

    @staticmethod
    def select_device_dtype(user_device: str = None, user_dtype: str = None) -> tuple[str, torch.dtype]:
        """Pick an available device (CUDA > MPS > CPU) and dtype.

        If a device or dtype is provided, they are honored. Otherwise we default
        to fp16 on accelerators (CUDA/MPS) and fp32 on CPU for compatibility.
        """
        if user_device:
            device = user_device
        else:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if user_dtype:
            map_dtype = {
                "fp32": torch.float32,
                "float32": torch.float32,
                "fp16": torch.float16,
                "float16": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
            }
            dtype = map_dtype.get(user_dtype.lower())
            if dtype is None:
                raise SystemExit(f"Unsupported dtype '{user_dtype}'. Use one of: fp32, fp16, bf16")
        else:
            if device.startswith("cuda") or device == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
        return device, dtype

    def load_openvla(self) -> None:
        """Load processor + model, trying multiple attention backends.

        On CUDA: flash_attention_2 → sdpa → eager; on others: sdpa → eager.
        Falls back to float32 on MPS if half-precision is unsupported.
        """
        self.require_timm_compatible()

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.device.startswith("cuda"):
            attn_order = ["flash_attention_2", "sdpa", "eager"]
        else:
            attn_order = ["sdpa", "eager"]
        last_err = None
        for attn in attn_order:
            try:
                kwargs = dict(load_kwargs)
                kwargs["attn_implementation"] = attn
                if self.local_model_dir:
                    proc = AutoProcessor.from_pretrained(self.local_model_dir, trust_remote_code=True,
                                                         local_files_only=True)
                    model = AutoModelForVision2Seq.from_pretrained(self.local_model_dir, **kwargs,
                                                                   local_files_only=True)
                else:
                    mid = self.model_id or os.environ.get("OPENVLA_MODEL_ID", "openvla/openvla-7b")
                    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True, token=self.hf_token)
                    model = AutoModelForVision2Seq.from_pretrained(mid, **kwargs, token=self.hf_token)
                try:
                    model = model.to(self.device, dtype=self.torch_dtype)
                    actual_dtype = self.torch_dtype
                except Exception as move_err:
                    if self.device == "mps" and self.torch_dtype == torch.float16:
                        print(f"Half precision on MPS failed ({move_err}); falling back to float32.")
                        model = model.to(self.device, dtype=torch.float32)
                        actual_dtype = torch.float32
                    else:
                        raise
                model.eval()
                print(f"Loaded model with attention implementation: {attn}")
                self.proc, self.model, self.torch_dtype = proc, model, actual_dtype

                # Load dataset statistics if available
                if DATASET_STATS_PATH and UN_NORMALIZED_DATASET_KEY:
                    if os.path.exists(DATASET_STATS_PATH):
                        with open(DATASET_STATS_PATH, "r") as f:
                            self.model.norm_stats[UN_NORMALIZED_DATASET_KEY] = json.load(f)
                            print(f"Loaded dataset statistics from: {DATASET_STATS_PATH}")
                    else:
                        print(f"No dataset statistics found at: {DATASET_STATS_PATH}")
                return
            except Exception as e:
                last_err = e
                print(f"Attention backend '{attn}' failed; trying next. Reason: {e}")
        raise RuntimeError(f"Failed to load model with available attention backends. Last error: {last_err}")

    @staticmethod
    def to_device_dtype(batch: dict, device: str, dtype: torch.dtype) -> dict:
        """Move a processed batch to the target device and dtype."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if v.dtype.is_floating_point:
                    v = v.to(dtype)
            out[k] = v
        return out

    def predict_action(self, img: Image.Image) -> List[float]:
        """Run a single RGB image through the model and normalize outputs.

        Returns a 7-dim action: [dx, dy, dz, droll, dpitch, dyaw, gripper].
        """
        assert self.proc is not None and self.model is not None, "Model not loaded. Call load_openvla() first."
        inputs = self.proc(text=self.prompt, images=img, return_tensors="pt")
        inputs = self.to_device_dtype(inputs, self.device, self.torch_dtype)
        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        if isinstance(action, torch.Tensor):
            if action.ndim > 1:
                action = action.squeeze(0)
            action = action.detach().cpu().float().tolist()
        elif np is not None and hasattr(np, 'ndarray') and isinstance(action, np.ndarray):
            if action.ndim > 1 and action.shape[0] == 1:
                action = action.squeeze(0)
            action = action.astype('float32').tolist()
        elif isinstance(action, (list, tuple)):
            action = list(map(float, action))
        elif isinstance(action, dict):
            for k in ("actions", "action", "pred", "prediction", "output"):
                if k in action:
                    v = action[k]
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().float()
                        if v.ndim > 1:
                            v = v.squeeze(0)
                        action = v.tolist()
                    elif np is not None and hasattr(np, 'ndarray') and isinstance(v, np.ndarray):
                        if v.ndim > 1 and v.shape[0] == 1:
                            v = v.squeeze(0)
                        action = v.astype('float32').tolist()
                    elif isinstance(v, (list, tuple)):
                        action = list(map(float, v))
                    break
        else:
            try:
                action = [float(action)]
            except Exception:
                raise TypeError(f"Unsupported action type {type(action)}; value={action}")
        if len(action) != 7:
            raise ValueError(f"Unexpected action length: {len(action)} (expected 7)")
        return [float(x) for x in action]

    def correct_gripper(self, action: List[float]) -> List[float]:
        """Apply per-dataset gripper correction (invert/binarize) if configured.

        Keeps the rest of the action unchanged. Returns a new list.
        """
        if not action:
            return action
        rules = DATASET_GRIPPER_RULES.get(self.unnorm_key, {})
        if not rules:
            return action
        out = list(action)
        try:
            g = float(out[-1])
            if rules.get("invert"):
                g = 1.0 - g
            if rules.get("binarize"):
                thr = float(rules.get("threshold", 0.5))
                g = 1.0 if g >= thr else 0.0
            out[-1] = g
        except Exception:
            # If conversion fails, leave unchanged
            return action
        return out

    def run_on_gif(self) -> None:
        """Iterate frames of self.gif_path, predict actions, log + save CSV."""
        if not os.path.exists(self.gif_path):
            raise SystemExit(f"GIF not found: {self.gif_path}")
        # Log execution context and load model once.
        self.info(f"Using device={self.device}, dtype={self.torch_dtype}")
        self.info("Loading OpenVLA model ...")
        self.load_openvla()
        self.info(f"Opening GIF: {self.gif_path}")
        gif = Image.open(self.gif_path)

        names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

        def fmt(vals: List[float]) -> str:
            return ", ".join(f"{k}={v:.6f}" for k, v in zip(names, vals))

        # Prepare CSV destination and write header.
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        frame_count = 0
        with open(self.csv_path, mode="w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["frame_id", *names])
            for frame_idx, frame in enumerate(ImageSequence.Iterator(gif)):
                # Convert each frame to RGB to avoid palette mode quirks.
                img = frame.convert("RGB")
                pred_raw = self.predict_action(img)
                pred = self.correct_gripper(pred_raw)
                self.info(f"Frame {frame_idx:04d} action: " + json.dumps([float(f"{x:.12f}") for x in pred]))
                #self.info(f"Frame {frame_idx:04d} action (labeled): " + fmt(pred))
                writer.writerow([frame_idx, *pred])
                frame_count += 1
        self.info(f"Processed {frame_count} frames. Wrote: {self.csv_path}. Done.")


def run_server(port: int = 8745, host: str = "127.0.0.1") -> None:
    """Start a local HTTP server that accepts POSTed images and returns JSON.

    Endpoints:
      - POST /predict: body contains image data (raw bytes), or multipart with
        a field named 'file', or JSON with a base64-encoded image under
        'image_base64'. Optional JSON field 'prompt' overrides the default.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    brain = RobotBrain()
    brain.info(f"Using device={brain.device}, dtype={brain.torch_dtype}")
    brain.info("Loading OpenVLA model for server mode ...")
    brain.load_openvla()

    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    class Handler(BaseHTTPRequestHandler):
        # Suppress default noisy logging; keep concise one-liners
        def log_message(self, format: str, *args) -> None:  # type: ignore[override]
            return

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # type: ignore[override]
            try:
                if self.path.rstrip("/") != "/predict":
                    self._send_json(404, {"ok": False, "error": "Not Found"})
                    return

                ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length <= 0:
                    self._send_json(400, {"ok": False, "error": "Empty request body"})
                    return

                img: Image.Image | None = None
                prompt_override: str | None = None

                if ctype.startswith("application/json"):
                    raw = self.rfile.read(length)
                    try:
                        body = json.loads(raw.decode("utf-8"))
                    except Exception as e:
                        self._send_json(400, {"ok": False, "error": f"Invalid JSON: {e}"})
                        return
                    if isinstance(body, dict):
                        b64 = body.get("image_base64")
                        if b64:
                            try:
                                img_bytes = base64.b64decode(b64)
                                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                            except Exception as e:
                                self._send_json(400, {"ok": False, "error": f"Invalid base64 image: {e}"})
                                return
                        prompt_override = body.get("prompt") or None

                elif ctype.startswith("multipart/form-data"):
                    env = {
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    }
                    fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=env)  # type: ignore[arg-type]
                    file_item = fs["file"] if "file" in fs else None
                    if file_item is not None and file_item.file:
                        try:
                            img = Image.open(file_item.file).convert("RGB")
                        except Exception as e:
                            self._send_json(400, {"ok": False, "error": f"Invalid image in form-data: {e}"})
                            return
                    if "prompt" in fs and getattr(fs["prompt"], "value", None):
                        prompt_override = fs["prompt"].value

                else:
                    # Assume raw bytes of the image
                    data = self.rfile.read(length)
                    try:
                        img = Image.open(BytesIO(data)).convert("RGB")
                    except Exception as e:
                        self._send_json(400, {"ok": False, "error": f"Invalid image bytes: {e}"})
                        return

                if img is None:
                    self._send_json(400, {"ok": False, "error": "No image provided"})
                    return

                # If a prompt override is supplied, temporarily use it
                orig_prompt = brain.prompt
                if prompt_override:
                    brain.prompt = str(prompt_override)
                try:
                    action_raw = brain.predict_action(img)
                    action = brain.correct_gripper(action_raw)
                finally:
                    brain.prompt = orig_prompt

                RobotBrain.info(action)
                self._send_json(200, action)
            except Exception as e:
                brain.info(f"[SERVER] Error: {e}")
                self._send_json(500, {"ok": False, "error": str(e)})

    server = ThreadingHTTPServer((host, port), Handler)
    brain.info(f"Server listening on http://{host}:{port} (POST /predict)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        brain.info("Shutting down server ...")
        server.server_close()


def main():
    if SERVER_MODE:
        run_server()
    else:
        brain = RobotBrain()
        brain.run_on_gif()


if __name__ == "__main__":
    main()
