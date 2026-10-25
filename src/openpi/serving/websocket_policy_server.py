import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

# ====== UAV HTTP 接口所需依赖 ======
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import threading

import numpy as np
from PIL import Image
# ====== UAV HTTP 接口依赖结束 ======

logger = logging.getLogger(__name__)


# ====== UAV HTTP 接口（与 websocket 并列） ======
# 说明：
#   - websockets 的 process_request 只能处理握手，读不到 POST body，
#     所以不能把 /predict 直接塞进 _handler；
#   - 这里在同一个进程里再开一个标准库 HTTP server，端口默认 5007，
#     可用环境变量 UAV_HTTP_PORT 覆盖，设为 0 则关闭。
class _UAVHTTPHandler(BaseHTTPRequestHandler):
    """UAV-Flow-Eval 使用的极简 HTTP 协议（仅 /predict 和 /reset）。"""

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path == "/reset":
                # 当前实现不保存 episode 首帧，也无需重置模型状态；
                # 保留该路由是为了兼容 UAV-Flow-Eval 的 reset_model() 调用。
                self._reply({"status": "success"})
                return

            if path != "/predict":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            actions, raw_chunk = self.server.owner._predict_http(data)  # type: ignore[attr-defined]

            self._reply({
                "status": "success",
                "action": actions,
                # π0 原始 body-frame delta，仅调试用；客户端不会读
                "action_ori": raw_chunk,
            })
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._reply({"status": "error", "message": str(exc)}, code=500)

    def _reply(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D102
        # 避免每个请求都刷访问日志
        pass


class _UAVHTTPServer(ThreadingHTTPServer):
    """把 WebsocketPolicyServer 作为 owner 传给 handler，便于复用同一个 policy。"""

    daemon_threads = True

    def __init__(self, addr, owner):
        self.owner = owner
        super().__init__(addr, _UAVHTTPHandler)
# ====== UAV HTTP 接口定义结束 ======


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        # ====== UAV HTTP 接口状态 ======
        # 默认监听 5007；UAV_HTTP_PORT=0 可关闭。
        self._http_port = int(os.environ.get("UAV_HTTP_PORT", "5007"))
        self._http_server: _UAVHTTPServer | None = None
        self._first_image: np.ndarray | None = None
        # HTTP 与 websocket 共用同一个 policy，串行化推理调用
        self._lock = threading.Lock()
        # ====== UAV HTTP 接口状态结束 ======

    def serve_forever(self) -> None:
        # ====== 启动 UAV HTTP 接口 ======
        # HTTP 线程是 daemon，主线程继续跑下面的 websocket server。
        if self._http_port > 0:
            try:
                self._http_server = _UAVHTTPServer((self._host, self._http_port), self)
                threading.Thread(
                    target=self._http_server.serve_forever,
                    daemon=True,
                    name="uav-http-server",
                ).start()
                logger.info("UAV HTTP server listening on %s:%s", self._host, self._http_port)
            except OSError:
                logger.exception("Failed to start UAV HTTP server on port %s", self._http_port)
        # ====== UAV HTTP 接口启动结束 ======

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
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                # ====== 与 HTTP 接口共用同一个 policy，串行化 ======
                with self._lock:
                    action = self._policy.infer(obs)
                # ====== 串行化结束 ======
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # ====== UAV HTTP 推理：body-frame delta -> 首帧局部绝对位姿 ======
    def _predict_http(self, data: dict) -> tuple[list[list[float]], list[list[float]]]:
        """处理一次 /predict 请求。

        Returns:
            actions:   [H, 4] 首帧局部绝对位姿 [[x, y, z, yaw_rad], ...]
            raw_chunk: [H, 4] π0 原始 body-frame delta（调试用）
        """
        # 1) base64(PNG) -> uint8 HWC
        raw = base64.b64decode(data["image"])
        image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)

        # 2) 评测端 proprio: [x, y, z, yaw_deg]
        proprio = np.asarray(data["proprio"], dtype=np.float32).reshape(-1)[:4]
        if proprio.size < 4:
            raise ValueError(f"proprio 至少需要 4 维 [x,y,z,yaw]，收到 {proprio.size} 维")

        # 3) state 通信/训练都用度：proprio 原样传给模型，不做 deg2rad。
        state = proprio.copy()

        # key 必须与 openpi/src/openpi/policies/uav_policy.py 的 UAVInputs 一致
        obs = {
            "observation/image": image,
            "observation/state": state,
            "prompt": data["instr"],
        }

        # ====== 临时：固定动作，仅用于测试 client 链路 ======
        # 语义：首帧局部绝对位姿 [x, y, z, yaw_rad]。
        # 删除下面这段 return 即可恢复真实推理。
        fixed_actions = [
            [10.0, 0.0, 1.0, float(np.pi / 2)],
            [10.0, 0.0, 1.0, float(np.pi / 2)],
        ]
        return fixed_actions, fixed_actions

        # 4) 推理。UAVOutputs 通常已截到 4 维，这里再截一次兼容 action_dim=32 的配置。
        with self._lock:
            result = self._policy.infer(obs)

        chunk = np.asarray(result["actions"], dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.ndim != 2:
            raise ValueError(f"policy 返回的 actions 形状异常: {chunk.shape}")
        chunk = chunk[:, :4]

        # 5) body-frame delta -> 首帧局部绝对位姿
        #    p_t   = p_{t-1} + R(yaw_{t-1}) @ [dx, dy, dz]
        #    yaw_t = wrap(yaw_{t-1} + dyaw)
        # 注意：action 的 dyaw 是弧度，所以积分时把 pose 的 yaw 转成弧度；
        # 返回的 actions yaw 也保持弧度，客户端会再 np.degrees 转成度。
        pose = proprio.astype(np.float64).copy()
        pose[3] = np.deg2rad(proprio[3])

        actions: list[list[float]] = []
        for dx, dy, dz, dyaw in chunk:
            yaw = pose[3]                       #  使用最新航向角rad
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            R = np.array([
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ])
            pose[:3] = pose[:3] + R @ np.array([dx, dy, dz], dtype=np.float64)
            pose[3] = (yaw + dyaw + np.pi) % (2.0 * np.pi) - np.pi      # 更新航向角rad
            actions.append([float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])])

        return actions, chunk.tolist()
    # ====== UAV HTTP 推理结束 ======


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
