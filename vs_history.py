"""행동 경계의 물리 상태를 CPU 배열로 기록한다. Genesis 객체 자체는 pickle하지 않는다."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch

SCHEMA = 1
MPM_FIELDS = ("pos", "vel", "C", "F", "Jp", "active")
CONTROLLER_FIELDS = ("last_q", "initial_q", "tool_quat_down", "dof_idx",
                     "visual_counter", "initialized", "_history_position_command")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def height2d(value):
    a = torch.as_tensor(value).detach()
    if a.ndim == 4 and a.shape[:2] == (1, 1):
        a = a[0, 0]
    elif a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if tuple(a.shape) != (512, 512):
        raise ValueError(f"512×512 원본 높이맵이 필요합니다. 현재: {tuple(a.shape)}")
    if not bool(torch.isfinite(a).all()):
        raise ValueError("높이맵에 NaN 또는 Inf가 있습니다.")
    return a


def scale_target_xy(value, scale, *, background=0.0, mode="bilinear"):
    """512×512 중심을 기준으로 XY만 축소한다. z값에 배율을 곱하지 않는다."""
    scale = float(scale)
    if not 0.1 <= scale <= 1.0:
        raise ValueError("목표 크기 배율은 0.1 이상 1.0 이하여야 합니다.")
    a = height2d(value)
    if scale == 1.0:
        return a.clone()
    if mode not in ("bilinear", "nearest"):
        raise ValueError("지원하지 않는 목표 보간 방식입니다.")
    import torch.nn.functional as F
    a = a.float()
    affine = a.new_tensor([[[1.0 / scale, 0., 0.], [0., 1.0 / scale, 0.]]])
    grid = F.affine_grid(affine, (1, 1, 512, 512), align_corners=True)
    return F.grid_sample((a-float(background))[None, None], grid,
                         mode=mode, padding_mode="zeros", align_corners=True)[0, 0] + float(background)


def border_background(value):
    a = height2d(value)
    return float(torch.cat((a[0], a[-1], a[:, 0], a[:, -1])).median().item())


def load_heightmap(path, device, dtype, spatial_scale=1.0):
    a = np.load(Path(path).expanduser(), allow_pickle=False)
    if a.shape != (512, 512) or a.dtype.kind not in "fi":
        raise ValueError("목표는 실수 또는 정수형 512×512 .npy 배열이어야 합니다.")
    if not np.isfinite(a).all() or np.max(np.abs(a)) > 1.0:
        raise ValueError("목표 높이맵의 유효값과 단위(m)를 확인하세요.")
    h = height2d(torch.as_tensor(a, device=device, dtype=dtype))
    return scale_target_xy(h, spatial_scale, background=border_background(h)).to(dtype=dtype)


def load_roi(path):
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.suffix.lower() == ".npy":
        a = np.load(p, allow_pickle=False)
    else:
        with Image.open(p) as im:
            a = np.asarray(im.convert("L"))
    if a.shape != (512, 512) or a.dtype.kind not in "bifu" or not np.isfinite(a).all():
        raise ValueError("ROI는 유효한 512×512 마스크여야 합니다. 자동 크기 변경은 하지 않습니다.")
    mask = a > 0
    if not mask.any() or mask.all():
        raise ValueError("ROI는 비어 있거나 전체 화면일 수 없습니다.")
    return mask


def cpu_copy(value):
    if torch.is_tensor(value):
        base = value.detach()
        if type(base) is not torch.Tensor:
            base = base.as_subclass(torch.Tensor)
        return base.cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"지원하지 않는 상태 값: {type(value)}")


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.random.get_rng_state().clone(),
                cuda=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [])


def restore_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"].cpu())
    if rng["cuda"]:
        if not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("저장 당시와 CUDA 장치 수가 다릅니다.")
        torch.cuda.set_rng_state_all(rng["cuda"])


def pack_physics(state):
    """프레임 객체의 텐서만 복사해 장면 참조와 GPU 저장소 공유를 제거한다."""
    packed = []
    found_mpm = False
    for part in state.solvers_state:
        if part is None:
            packed.append(None)
            continue
        tensors = {k: cpu_copy(v) for k, v in vars(part).items() if torch.is_tensor(v)}
        if not tensors:
            raise RuntimeError(f"지원하지 않는 활성 solver 상태: {type(part).__name__}")
        is_mpm = all(hasattr(part, k) for k in MPM_FIELDS)
        if is_mpm:
            found_mpm = True
            # 속성 이름이 내부 저장 이름과 달라도 필요한 MPM 텐서는 반드시 들어가야 한다.
            for name in MPM_FIELDS:
                value = getattr(part, name)
                if not torch.is_tensor(value) or not any(
                    value is v for v in vars(part).values()
                ):
                    raise RuntimeError(f"MPM {name} 상태 저장 방식을 확인해야 합니다.")
        packed.append(dict(type=type(part).__name__, tensors=tensors, is_mpm=is_mpm))
    if not found_mpm:
        raise RuntimeError("전체 장면 상태에서 MPM 필드 pos/vel/C/F/Jp/active를 찾지 못했습니다.")
    return packed


@torch.no_grad()
def unpack_physics(state, packed):
    if len(state.solvers_state) != len(packed):
        raise RuntimeError("solver 구성이 저장 당시와 다릅니다.")
    # 쓰기 전에 모든 이름·형상·자료형부터 검사한다.
    writes = []
    for part, saved in zip(state.solvers_state, packed):
        if saved is None:
            if part is not None:
                raise RuntimeError("활성 solver 구성이 저장 당시와 다릅니다.")
            continue
        if part is None or type(part).__name__ != saved["type"]:
            raise RuntimeError("solver 상태 형식이 저장 당시와 다릅니다.")
        live_fields = {k for k, v in vars(part).items() if torch.is_tensor(v)}
        if live_fields != set(saved["tensors"]):
            raise RuntimeError("solver 텐서 필드가 저장 당시와 다릅니다.")
        for name, value in saved["tensors"].items():
            dest = getattr(part, name)
            if dest.shape != value.shape or dest.dtype != value.dtype:
                raise RuntimeError(f"상태 형상/자료형 불일치: {saved['type']}.{name}")
            writes.append((dest, value))
    for dest, value in writes:
        dest.copy_(value.to(device=dest.device))
    return state


def json_write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def tensor_save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def physics_signature(module):
    """장면 생성에 쓰인 코드·설정·URDF/메시·버전을 한 번 해시한다."""
    import vs_config as config
    root = Path(module.__file__).resolve().parent
    names = ("vs_config.py", "vs_simulator.py", "vs_representation.py", "vs_history.py")
    sources = {n: sha256(root / n) for n in names}
    settings = {}
    for name, value in vars(config).items():
        if name.isupper() and not name.endswith(("DIR", "PATH")) and name != "SHOW_VIEWER":
            if isinstance(value, (str, bool, int, float, tuple, list, dict)):
                settings[name] = value
    assets = {}
    for name in ("BASE_IIWA_URDF", "ROBOT_URDF"):
        p = Path(getattr(module, name))
        assets[name] = sha256(p)
        for elem in ET.parse(p).getroot().iter("mesh"):
            filename = elem.get("filename")
            if not filename or "://" in filename:
                raise RuntimeError(f"메시 경로를 해시할 수 없습니다: {filename}")
            mesh = Path(filename)
            if not mesh.is_absolute():
                mesh = p.parent / mesh
            assets[f"{name}:{filename}"] = sha256(mesh)
    try:
        genesis_version = importlib.metadata.version("genesis-world")
    except importlib.metadata.PackageNotFoundError:
        import genesis
        genesis_version = str(getattr(genesis, "__version__", "unknown"))
    signature = dict(sources=sources, settings=settings, assets=assets,
                     material=os.environ.get("VS_MATERIAL", "medium"),
                     tip_correction=bool(module.TIP_CORRECTION_ENABLED),
                     genesis=genesis_version, torch=str(torch.__version__), numpy=np.__version__,
                     cuda=torch.version.cuda,
                     gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    # JSON 왕복으로 tuple/list 표현 차이도 제거한다.
    signature = json.loads(json.dumps(signature, sort_keys=True))
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return dict(digest=digest, details=signature)


class GenesisStateAdapter:
    def __init__(self, module):
        self.module = module
        self.simulator = module.initialize()
        self.scene = self.simulator.scene
        if not callable(getattr(self.scene, "get_state", None)) or not callable(getattr(self.scene, "reset", None)):
            raise RuntimeError("Genesis scene.get_state/reset API가 필요합니다.")

    def observe(self):
        return height2d(self.module.get_height_gpu()).cpu().clone()

    def capture(self):
        sim = self.simulator
        missing = [n for n in CONTROLLER_FIELDS if not hasattr(sim, n)]
        if missing:
            raise RuntimeError(f"제어기 추적 필드가 없습니다: {missing}")
        return dict(schema=SCHEMA, physics=pack_physics(self.scene.get_state()),
                    controller={n: cpu_copy(getattr(sim, n)) for n in CONTROLLER_FIELDS},
                    height=self.observe(), rng=capture_rng(),
                    robot=dict(q=cpu_copy(sim.robot.get_dofs_position(dofs_idx_local=sim.dof_idx)),
                               velocity=cpu_copy(sim.robot.get_dofs_velocity(dofs_idx_local=sim.dof_idx)),
                               tip_pos=cpu_copy(sim.tip.get_pos()), tip_quat=cpu_copy(sim.tip.get_quat())))

    def restore(self, saved):
        if saved["schema"] != SCHEMA:
            raise RuntimeError("지원하지 않는 스냅샷 형식입니다.")
        state = unpack_physics(self.scene.get_state(), saved["physics"])
        self.scene.reset(state=state)
        for name, value in saved["controller"].items():
            live = getattr(self.simulator, name)
            value = value.to(device=live.device, dtype=live.dtype).clone() if torch.is_tensor(live) else cpu_copy(value)
            setattr(self.simulator, name, value)
        # baseline에서 사용하는 위치 제어 명령도 복원한다. 물리 step은 진행하지 않는다.
        self.simulator.control_arm_position(
            position=self.simulator._history_position_command,
            dofs_idx_local=self.simulator.dof_idx,
        )
        restore_rng(saved["rng"])
        return self.observe()

    def execute(self, action, action_index):
        return self.module.linear_push(action, action_index=action_index)


def error_metrics(before, after, predicted, target, roi):
    b, a, p, g = (height2d(x).cpu().float() for x in (before, after, predicted, target))
    d = a - b
    result = dict(global_before_mae_mm=float((b-g).abs().mean()) * 1000,
                  global_after_mae_mm=float((a-g).abs().mean()) * 1000,
                  model_gap_mae_mm=float((p-a).abs().mean()) * 1000,
                  deformation_mae_mm=float(d.abs().mean()) * 1000)
    if roi is not None:
        r = torch.as_tensor(roi, dtype=torch.bool)
        result.update(roi_before_mae_mm=float((b-g)[r].abs().mean()) * 1000,
                      roi_after_mae_mm=float((a-g)[r].abs().mean()) * 1000,
                      roi_deformation_mae_mm=float(d[r].abs().mean()) * 1000,
                      outside_deformation_max_mm=float(d[~r].abs().max()) * 1000,
                      outside_target_mae_mm=float((a-g)[~r].abs().mean()) * 1000)
    return result


class ActionHistory:
    def __init__(self, module, directory, *, target, raw_target, roi_path=None,
                 target_path=None, checkpoint_path=None, target_transform=None, run_args=None):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"이력 폴더가 비어 있지 않습니다: {self.directory}")
        self.adapter = GenesisStateAdapter(module)
        self.target = height2d(target).cpu().clone()
        self.roi = load_roi(roi_path)
        spatial_scale = float((target_transform or {}).get("spatial_scale", 1.0))
        if self.roi is not None and spatial_scale != 1.0:
            self.roi = scale_target_xy(torch.from_numpy(self.roi).float(), spatial_scale,
                                       mode="nearest").numpy() > 0.5
            if not self.roi.any():
                raise ValueError("축소 후 귀 ROI가 비었습니다.")
        self.count = 0
        self.signature = physics_signature(module)
        np.save(self.directory / "target_raw_512_m.npy", height2d(raw_target).cpu().numpy())
        if target_path is not None:
            np.save(self.directory / "target_source_original_512_m.npy",
                    np.load(Path(target_path).expanduser(), allow_pickle=False))
        np.save(self.directory / "target_used_512_m.npy", self.target.numpy())
        if self.roi is not None:
            np.save(self.directory / "roi_mask_512.npy", self.roi)
        if target_transform is not None:
            np.save(self.directory / "target_edge_mask_512.npy", target_transform["edge_mask"].cpu().numpy())
            np.save(self.directory / "target_edge_reference_512_m.npy", target_transform["edge_reference"].cpu().numpy())
            transform = {k: v for k, v in target_transform.items() if not torch.is_tensor(v)}
        else:
            transform = None
        def file_info(p):
            return None if p is None else dict(path=str(Path(p).expanduser().resolve()), sha256=sha256(Path(p).expanduser()))
        import vs_config as config
        manifest = dict(schema=SCHEMA, signature=self.signature, run_args=run_args,
                        units=dict(height="m", xy="m", theta="rad", length="m", depth="m"),
                        shape=[512, 512], state_indexing="action_000001: S0 -> a0 -> S1",
                        target_source=file_info(target_path), roi_source=file_info(roi_path),
                        checkpoint=file_info(checkpoint_path), target_transform=transform,
                        static_files={p.name: sha256(p) for p in self.directory.glob("*.npy")},
                        map_axes=dict(column="world +x", row="world +y", resampling=False,
                                      x_bounds=[config.CLAY_CENTER_X-config.CLAY_SIZE_X/2, config.CLAY_CENTER_X+config.CLAY_SIZE_X/2],
                                      y_bounds=[config.CLAY_CENTER_Y-config.CLAY_SIZE_Y/2, config.CLAY_CENTER_Y+config.CLAY_SIZE_Y/2]),
                        replay_validated=False,
                        notes=["512×512는 입자에서 직접 관측",
                               "scene.reset은 시뮬레이션 시계를 초기화함. baseline에는 시간 의존 외력이 없음.",
                               "엔진 내부 난수 및 미노출 solver 캐시는 replay drift 실측으로 검증해야 함."])
        json_write(self.directory / "manifest.json", manifest)

    def record_push(self, action, *, action_index, before, predicted, planner_context=None):
        # snapshot은 본 행동 직전, 실행이 시작되기 전에 디스크에 확정한다.
        t = self.count
        if action_index != t:
            raise ValueError(f"행동 번호가 연속적이지 않습니다: expected={t}, got={action_index}")
        a = torch.as_tensor(action).detach().reshape(-1).cpu().clone()
        if a.numel() != 5 or not bool(torch.isfinite(a).all()):
            raise ValueError("행동은 유효한 [x,y,theta,length,depth] 벡터여야 합니다.")
        prefix = f"action_{t+1:06d}"
        pending = self.directory / (prefix + ".incomplete")
        pending.mkdir()
        pre = self.adapter.capture()
        before = height2d(before).cpu().clone()
        if not torch.allclose(before, pre["height"], atol=1e-7, rtol=0):
            raise RuntimeError("계획기의 행동 전 관측과 스냅샷 높이맵이 다릅니다.")
        tensor_save(pending / "pre_state.pt", pre)
        tensor_save(pending / "planner_context.pt", cpu_copy(planner_context or {}))
        json_write(pending / "pending_action.json", dict(t=t, action=a.tolist(), action_index=action_index))
        start = time.perf_counter()
        result = self.adapter.execute(a, action_index)
        execution_seconds = time.perf_counter() - start
        post = self.adapter.capture()
        after = post["height"]
        pred = height2d(predicted).cpu().clone()
        tensor_save(pending / "post_state.pt", post)
        np.savez_compressed(pending / "transition.npz", action=a.numpy(),
                            before_m=before.numpy(), after_m=after.numpy(),
                            delta_m=(after-before).numpy(), predicted_after_m=pred.numpy(),
                            predicted_delta_m=(pred-before).numpy())
        record = dict(schema=SCHEMA, t=t, action_number=t+1, action_index=action_index,
                      action=a.tolist(), execution_seconds=execution_seconds,
                      metrics=error_metrics(before, after, pred, self.target, self.roi))
        json_write(pending / "record.json", record)
        hashes = {p.name: sha256(p) for p in pending.iterdir() if p.is_file()}
        json_write(pending / "checksums.json", hashes)
        pending.rename(self.directory / prefix)
        self.count += 1
        print(f"[HISTORY {t+1:06d}] 전체 MPM·로봇·제어기·난수 상태 저장 완료")
        return result


def verify_action_files(directory):
    directory = Path(directory)
    checks = json.loads((directory / "checksums.json").read_text())
    required = {"pre_state.pt", "post_state.pt", "planner_context.pt", "transition.npz", "record.json", "pending_action.json"}
    if set(checks) != required:
        raise RuntimeError(f"이력 파일 목록이 잘못됐습니다: {directory}")
    for name, expected in checks.items():
        if sha256(directory / name) != expected:
            raise RuntimeError(f"이력 파일 해시 불일치: {directory / name}")


def load_snapshot(path):
    # 이 도구가 직접 생성한 상태 파일만 대상으로 한다.
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_saved_target_transform(raw, history_dir):
    """축소 전 원본 H_edit에 조형 때의 XY 배율/alpha/beta/경계를 함께 적용한다."""
    root = Path(history_dir)
    meta = json.loads((root / "manifest.json").read_text())["target_transform"]
    if meta is None:
        raise ValueError("저장된 목표 변환이 없습니다.")
    a = height2d(raw).cpu().float()
    a = scale_target_xy(a, meta.get("spatial_scale", 1.0),
                        background=meta.get("spatial_background_m", 0.0))
    edge = torch.from_numpy(np.load(root / "target_edge_mask_512.npy", allow_pickle=False)).bool()
    reference = torch.from_numpy(np.load(root / "target_edge_reference_512_m.npy", allow_pickle=False))
    out = torch.where(edge, reference, meta["alpha"] * a + meta["beta_m"])
    return out.clamp(meta["clip_min_m"], meta["clip_max_m"])
