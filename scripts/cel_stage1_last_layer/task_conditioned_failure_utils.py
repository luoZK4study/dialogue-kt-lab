from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path


FAILURE_PATTERNS = [
    ("oom", re.compile(r"OutOfMemoryError|CUDA out of memory|torch\.OutOfMemoryError", re.I)),
    ("traceback", re.compile(r"Traceback \(most recent call last\):")),
]

PROGRESS_RE = re.compile(r"(Training|Validation|Validating|Testing):\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)")
ASSERTION_RE = re.compile(r"(Assertion `.*?failed\.)")
LOSS_SITE_RE = re.compile(r"(\.\./aten/src/ATen/native/cuda/Loss\.cu:\d+:)")
TQDM_TIMING_RE = re.compile(
    r"(?P<elapsed>\d+:\d{2}(?::\d{2})?)<(?P<remaining>\d+:\d{2}(?::\d{2})?),\s*(?P<rate>[0-9.]+)(?P<unit>it/s|s/it)"
)
TIMESTAMPED_EVENT_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} [0-9:]+)\]\s+(.*)$")
MONITOR_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} [0-9:]+)$")
CONTROLLER_SLEEP_EVENT_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} [0-9:]+)\]\s+controller sleeping (\d+)s\b")


def _parse_hms(text: str | None) -> int | None:
    if not text:
        return None
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return None


def parse_tqdm_timing(progress_line: str | None) -> dict | None:
    if not progress_line:
        return None
    match = TQDM_TIMING_RE.search(progress_line)
    if not match:
        return None

    elapsed_secs = _parse_hms(match.group("elapsed"))
    remaining_secs = _parse_hms(match.group("remaining"))
    rate_value = float(match.group("rate"))
    unit = match.group("unit")
    seconds_per_iter = rate_value if unit == "s/it" else (None if rate_value <= 0 else 1.0 / rate_value)
    return {
        "elapsed_secs": elapsed_secs,
        "remaining_secs": remaining_secs,
        "seconds_per_iter": seconds_per_iter,
    }


def parse_timestamped_event_time(event: str | None) -> datetime | None:
    if event is None:
        return None
    match = TIMESTAMPED_EVENT_RE.match(event)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_monitor_timestamp(text: str | None) -> datetime | None:
    if not text:
        return None
    match = MONITOR_TIMESTAMP_RE.match(text.strip())
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def recommended_next_monitor_after(
    refresh_event: str | None,
    poll_interval: tuple[int | None, str] | None,
) -> datetime | None:
    if refresh_event is None or poll_interval is None:
        return None
    interval_secs, _reason = poll_interval
    if interval_secs is None:
        return None
    event_time = parse_timestamped_event_time(refresh_event)
    if event_time is None:
        return None
    return event_time + timedelta(seconds=interval_secs)


def format_monitor_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S")


def controller_next_cycle_estimate(event: str | None) -> datetime | None:
    if event is None:
        return None
    match = CONTROLLER_SLEEP_EVENT_RE.match(event.strip())
    if not match:
        return None
    try:
        event_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return event_time + timedelta(seconds=int(match.group(2)))


def format_wait_duration(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    if seconds <= 0:
        return "0s"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def format_seconds_per_iter(seconds_per_iter: float | None) -> str | None:
    if seconds_per_iter is None:
        return None
    return f"{seconds_per_iter:.2f}s/it"


def monitor_due_state(next_monitor_after_text: str | None, now: datetime | None = None) -> dict | None:
    next_monitor_after = parse_monitor_timestamp(next_monitor_after_text)
    if next_monitor_after is None:
        return None
    current_time = now or datetime.now()
    delta_secs = int((next_monitor_after - current_time).total_seconds())
    due_now = delta_secs <= 0
    return {
        "due_now": due_now,
        "seconds_until_due": 0 if due_now else delta_secs,
        "seconds_past_due": 0 if delta_secs >= 0 else abs(delta_secs),
        "remaining_wait_text": "0s" if due_now else format_wait_duration(delta_secs),
    }


def progress_timing_note(progress: dict | None, *, language: str) -> str | None:
    if progress is None:
        return None
    timing = progress.get("timing") or {}
    elapsed_text = format_wait_duration(timing.get("elapsed_secs"))
    remaining_text = format_wait_duration(timing.get("remaining_secs"))
    rate_text = format_seconds_per_iter(timing.get("seconds_per_iter"))
    if elapsed_text is None and remaining_text is None and rate_text is None:
        return None

    phase = str(progress.get("phase") or "").lower()
    if language == "cn":
        phase_prefix = {
            "training": "当前训练阶段计时：",
            "validation": "当前验证阶段计时：",
            "validating": "当前验证阶段计时：",
            "testing": "当前测试阶段计时：",
        }.get(phase, "当前阶段计时：")
        parts = []
        if elapsed_text is not None:
            parts.append(f"已耗时 `{elapsed_text}`")
        if remaining_text is not None:
            parts.append(f"剩余 ETA `{remaining_text}`")
        if rate_text is not None:
            parts.append(f"约 `{rate_text}`")
        return phase_prefix + "，".join(parts) + "。"

    phase_prefix = {
        "training": "Current training-phase timing: ",
        "validation": "Current validation-phase timing: ",
        "validating": "Current validation-phase timing: ",
        "testing": "Current testing-phase timing: ",
    }.get(phase, "Current phase timing: ")
    parts = []
    if elapsed_text is not None:
        parts.append(f"elapsed `{elapsed_text}`")
    if remaining_text is not None:
        parts.append(f"remaining ETA `{remaining_text}`")
    if rate_text is not None:
        parts.append(f"about `{rate_text}`")
    return phase_prefix + ", ".join(parts) + "."


def epoch_cycle_note(meta: dict | None, progress: dict | None, *, language: str) -> str | None:
    if meta is None:
        return None
    multistage_contract = meta.get("multistage_contract") or []
    if multistage_contract:
        stages = []
        for stage in multistage_contract:
            label = str(stage.get("label") or "stage")
            epochs = (stage.get("expected_args") or {}).get("epochs")
            stages.append(f"{label} ({epochs} epoch{'s' if str(epochs) != '1' else ''})")
        stage_text = " -> ".join(stages)
        if language == "cn":
            return (
                f"该候选是独立进程串行的多阶段链：`{stage_text}`；每个阶段的 epoch 计数会重新从 1 开始。"
                "bootstrap -> calibrator warmup -> joint 的阶段切换不是 run 重启；只有 joint 执行正式 test。"
            )
        return (
            f"This candidate is a serial multi-process chain: `{stage_text}`; each stage restarts its epoch counter at 1. "
            "The bootstrap -> calibrator warmup -> joint transition is not a run restart, and only joint performs the formal test."
        )
    planned_epochs = meta.get("planned_epochs")
    if not isinstance(planned_epochs, int) or planned_epochs <= 1:
        return None

    phase = str((progress or {}).get("phase") or "").lower()
    epoch = (progress or {}).get("epoch")

    if language == "cn":
        if epoch is None:
            return (
                f"该候选脚本配置 `--epochs {planned_epochs}`；epoch 间出现 `validation -> training` 的回切"
                "属于正常的多 epoch full-train 流程，不表示 run 重启。"
            )
        if phase in {"validation", "validating"} and epoch < planned_epochs:
            return (
                f"该候选脚本配置 `--epochs {planned_epochs}`；当前是 epoch {epoch} 的验证阶段，"
                f"完成后会继续进入 epoch {epoch + 1} training，这属于正常的 full-train 流程。"
            )
        if phase == "training" and epoch > 1:
            return (
                f"该候选脚本配置 `--epochs {planned_epochs}`；当前已进入 epoch {epoch} training，"
                "前一轮 `validation -> training` 回切属于正常的多 epoch full-train 流程，不表示 run 重启。"
            )
        return None

    if epoch is None:
        return (
            f"This candidate is configured with `--epochs {planned_epochs}`; a `validation -> training` "
            "phase return between epochs is normal multi-epoch full-train behavior, not a restart."
        )
    if phase in {"validation", "validating"} and epoch < planned_epochs:
        return (
            f"This candidate is configured with `--epochs {planned_epochs}`; the current validation phase for "
            f"epoch {epoch} should return to epoch {epoch + 1} training next, which is normal full-train behavior."
        )
    if phase == "training" and epoch > 1:
        return (
            f"This candidate is configured with `--epochs {planned_epochs}`; the current epoch {epoch} training "
            "phase is a normal post-validation return within a multi-epoch full-train run, not a restart."
        )
    return None


def recommend_round3_poll_interval(progress: dict | None, next_action: str | None) -> tuple[int, str] | None:
    if next_action not in {"wait_round3", "launch_round3"}:
        return None
    if progress is None:
        return 600, "default training cadence"

    phase = str(progress.get("phase") or "").lower()
    pct = progress.get("pct")
    total = progress.get("total")
    timing = progress.get("timing") or {}
    remaining_secs = timing.get("remaining_secs")
    seconds_per_iter = timing.get("seconds_per_iter")

    if remaining_secs is not None and remaining_secs <= 5 * 60:
        return 120, "<=5m remaining"
    if phase in {"validation", "validating", "testing"}:
        return 300, f"{phase} phase"
    if pct is not None and pct >= 95:
        return 300, ">=95% progress"
    if remaining_secs is not None and remaining_secs <= 30 * 60:
        return 300, "<=30m remaining"
    if remaining_secs is not None and remaining_secs <= 45 * 60:
        return 600, "<=45m remaining"

    if phase == "training":
        if pct is not None and pct < 10:
            if remaining_secs is not None and remaining_secs >= 90 * 60:
                return 900, "early training with long ETA"
            if total is not None and total >= 8000:
                return 900, "early large full-train run"
        if seconds_per_iter is not None and seconds_per_iter >= 1.5:
            return 600, "slow training iterations"
        return 600, "steady training cadence"

    return 600, "default training cadence"


def read_log_excerpt(path: Path, head_bytes: int = 4096, tail_bytes: int = 262144) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(min(head_bytes, size))
        if size > tail_bytes:
            f.seek(max(0, size - tail_bytes))
        tail = f.read()
    if size <= head_bytes:
        blob = head
    else:
        blob = head + b"\n" + tail
    return blob.decode("utf-8", errors="ignore").replace("\r", "\n")


def latest_phase_progress_from_text(text: str) -> dict | None:
    matches = list(PROGRESS_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    phase, pct, step, total = last.groups()
    line_start = text.rfind("\n", 0, last.start()) + 1
    line_end = text.find("\n", last.end())
    if line_end == -1:
        line_end = len(text)
    progress_line = text[line_start:line_end]
    epoch_matches = list(EPOCH_RE.finditer(text[: last.start()]))
    epoch = int(epoch_matches[-1].group(1)) if epoch_matches else None
    return {
        "phase": phase.lower(),
        "progress": f"{step}/{total} ({pct}%)",
        "epoch": epoch,
        "pct": int(pct),
        "step": int(step),
        "total": int(total),
        "timing": parse_tqdm_timing(progress_line),
        "progress_line": progress_line.strip(),
    }


def latest_phase_progress_from_log(path: Path) -> dict | None:
    text = read_log_excerpt(path)
    if not text:
        return None
    return latest_phase_progress_from_text(text)


def detect_failure_reason(text: str) -> str | None:
    for label, pattern in FAILURE_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _clean_failure_text(text: str | None) -> str | None:
    if text is None:
        return None
    return text.replace("`", "'").strip()


def extract_failure_evidence(path: Path) -> dict:
    text = read_log_excerpt(path)
    if not text:
        return {
            "failure_reason": None,
            "progress": None,
            "assertion_line": None,
            "runtime_line": None,
            "traceback_line": None,
        }

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    raw_assertion_line = next(
        (line for line in reversed(lines) if "Assertion `" in line and " failed" in line),
        None,
    )
    runtime_line = next((line for line in reversed(lines) if line.startswith("RuntimeError:")), None)
    traceback_line = next(
        (line for line in reversed(lines) if line.startswith("Traceback (most recent call last):")),
        None,
    )
    assertion_line = None
    assertion_site = None
    if raw_assertion_line is not None:
        site_match = LOSS_SITE_RE.search(raw_assertion_line)
        if site_match:
            assertion_site = _clean_failure_text(site_match.group(1))
        assertion_match = ASSERTION_RE.search(raw_assertion_line)
        if assertion_match:
            assertion_line = _clean_failure_text(assertion_match.group(1))
        else:
            assertion_line = _clean_failure_text(raw_assertion_line)
    return {
        "failure_reason": detect_failure_reason(text),
        "progress": latest_phase_progress_from_text(text),
        "assertion_line": assertion_line,
        "assertion_site": assertion_site,
        "runtime_line": _clean_failure_text(runtime_line),
        "traceback_line": _clean_failure_text(traceback_line),
    }


def format_progress_cn(progress: dict | None) -> str | None:
    if progress is None:
        return None
    epoch_text = f"epoch {progress['epoch']} " if progress.get("epoch") is not None else ""
    return f"{epoch_text}{progress['phase']} 约 `{progress['progress']}`"


def format_progress_en(progress: dict | None) -> str | None:
    if progress is None:
        return None
    epoch_text = f"epoch {progress['epoch']} " if progress.get("epoch") is not None else ""
    return f"{epoch_text}{progress['phase']} near `{progress['progress']}`"


def format_reference_progress_cn(reference: dict | None) -> str | None:
    if reference is None:
        return None
    epoch = reference.get("epoch")
    phase = reference.get("phase")
    step = reference.get("step")
    total = reference.get("total")
    pct = reference.get("pct")
    if phase is None or step is None or total is None or pct is None:
        return None
    epoch_text = f"epoch {epoch} " if epoch is not None else ""
    return f"{epoch_text}{phase} 约 `{step}/{total} ({pct}%)`"


def format_reference_progress_en(reference: dict | None) -> str | None:
    if reference is None:
        return None
    epoch = reference.get("epoch")
    phase = reference.get("phase")
    step = reference.get("step")
    total = reference.get("total")
    pct = reference.get("pct")
    if phase is None or step is None or total is None or pct is None:
        return None
    epoch_text = f"epoch {epoch} " if epoch is not None else ""
    return f"{epoch_text}{phase} near `{step}/{total} ({pct}%)`"


def progress_vs_reference(progress: dict | None, reference: dict | None) -> str | None:
    if progress is None or reference is None:
        return None
    if str(progress.get("phase") or "").lower() != str(reference.get("phase") or "").lower():
        return None
    reference_epoch = reference.get("epoch")
    if reference_epoch is not None and progress.get("epoch") != reference_epoch:
        return None
    reference_total = reference.get("total")
    if reference_total is not None and progress.get("total") != reference_total:
        return None
    current_step = progress.get("step")
    reference_step = reference.get("step")
    if current_step is None or reference_step is None:
        return None
    if current_step > reference_step:
        return "passed"
    if current_step == reference_step:
        return "reached"
    return None


def stability_milestone_note(meta: dict | None, progress: dict | None, *, language: str) -> str | None:
    if meta is None:
        return None
    reference = meta.get("prior_failure_progress")
    relation = progress_vs_reference(progress, reference)
    if relation is None:
        return None

    if language == "cn":
        reference_text = format_reference_progress_cn(reference)
        if reference_text is None:
            return None
        if relation == "passed":
            return (
                f"重跑稳定性里程碑：当前 active run 已越过上一轮同步到的失败点，约为 {reference_text}；"
                "当前 BCE 稳定性修复到旧崩溃区间为止仍然成立。"
            )
        return (
            f"重跑稳定性里程碑：当前 active run 已到达上一轮同步到的失败点，约为 {reference_text}；"
            "接下来继续观察能否稳定穿过旧崩溃区间。"
        )

    reference_text = format_reference_progress_en(reference)
    if reference_text is None:
        return None
    if relation == "passed":
        return (
            f"Rerun stability milestone: this active run has passed the prior synced failure point {reference_text}; "
            "the current BCE stabilization fix has held through the old crash region so far."
        )
    return (
        f"Rerun stability milestone: this active run has reached the prior synced failure point {reference_text}; "
        "keep watching whether it can clear the old crash region."
    )
