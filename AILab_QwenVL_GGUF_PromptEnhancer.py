# ComfyUI-QwenVL GGUF prompt enhancer
#
# GGUF nodes powered by llama.cpp for Qwen-VL models, including Qwen3-VL and Qwen2.5-VL.
# Provides vision-capable GGUF inference and prompt execution.
#
# Models are loaded via llama-cpp-python and configured through gguf_models.json.
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, hf_hub_url, snapshot_download
from huggingface_hub.utils import enable_progress_bars
from llama_cpp import Llama

import folder_paths
from AILab_OutputCleaner import OutputCleanConfig, clean_model_output
from comfy.utils import ProgressBar

try:
    import requests
except ImportError:
    requests = None

try:
    from huggingface_hub import get_token as _hf_get_token
except Exception:
    try:
        from huggingface_hub.utils import get_token as _hf_get_token
    except Exception:
        _hf_get_token = None


def _resolve_hf_token() -> str | None:
    if _hf_get_token is not None:
        try:
            token = _hf_get_token()
            if token:
                return token
        except Exception:
            pass
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

NODE_DIR = Path(__file__).parent
GGUF_CONFIG_PATH = NODE_DIR / "gguf_models.json"
PROMPT_CONFIG_PATH = NODE_DIR / "AILab_System_Prompts.json"


def load_prompt_config():
    if not PROMPT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"[QwenVL] Missing AILab_System_Prompts.json at {PROMPT_CONFIG_PATH}")
    try:
        with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwen_text = data.get("qwen_text") or {}
        styles = qwen_text.get("styles")
        translation_prompt = qwen_text.get("translation_prompt")
        if not styles or not translation_prompt:
            raise ValueError("AILab_System_Prompts.json must include qwen_text.styles and qwen_text.translation_prompt")
        return {"styles": styles, "translation_prompt": translation_prompt}
    except Exception as exc:
        raise RuntimeError(f"[QwenVL] Failed to load AILab_System_Prompts.json: {exc}") from exc


PROMPT_CONFIG = load_prompt_config()
STYLES = PROMPT_CONFIG.get("styles", {})


def _safe_dirname(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    return "".join(ch for ch in value if ch.isalnum() or ch in "._- ").strip() or "unknown"


def _resolve_base_dir(base_dir_value: str) -> Path:
    base_dir = Path(base_dir_value)
    if base_dir.is_absolute():
        return base_dir
    return Path(folder_paths.models_dir) / base_dir


def _model_name_to_filename_candidates(model_name: str) -> set[str]:
    raw = (model_name or "").strip()
    if not raw:
        return set()
    candidates = {raw, f"{raw}.gguf"}
    if " / " in raw:
        tail = raw.split(" / ", 1)[1].strip()
        candidates.update({tail, f"{tail}.gguf"})
    if "/" in raw:
        tail = raw.rsplit("/", 1)[-1].strip()
        candidates.update({tail, f"{tail}.gguf"})
    return candidates


def _format_download_progress(done: int, total: int, width: int = 22) -> str:
    total = max(total, 1)
    done = max(0, min(done, total))
    ratio = done / total
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}% ({done}/{total})"


def _stream_hf_file_with_percent(
    repo_id: str,
    filename: str,
    target_path: Path,
    chunk_size: int = 1024 * 1024,
):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return

    if requests is None:
        enable_progress_bars()
        print(f"[QwenVL] requests not available, using hf_hub_download for {filename}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(target_path.parent),
            local_dir_use_symlinks=False,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.replace(target_path)
        return

    token = _resolve_hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="model")
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    def _fmt_eta(seconds: float) -> str:
        seconds = max(int(seconds), 0)
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours:d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    file_label = Path(filename).name
    use_inline = bool(getattr(sys.stdout, "isatty", lambda: False)())
    last_render_len = 0

    def _emit_progress(message: str, done: bool = False):
        nonlocal last_render_len
        if use_inline:
            pad = " " * max(0, last_render_len - len(message))
            print(f"\r{message}{pad}", end="\n" if done else "", flush=True)
            last_render_len = 0 if done else len(message)
            return
        print(message)

    try:
        with requests.get(url, headers=headers, stream=True, timeout=(15, 600)) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("Content-Length") or response.headers.get("content-length") or 0)
            file_pbar = ProgressBar(1000)
            downloaded = 0
            last_bucket = -1
            last_log_time = 0.0
            total_mb = total_bytes / (1024 * 1024) if total_bytes > 0 else 0.0
            start_time = time.time()
            last_sample_time = start_time
            ema_speed_mbs = 0.0
            ema_alpha = 0.2

            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    delta_t = max(now - last_sample_time, 1e-6)
                    inst_speed_mbs = (len(chunk) / (1024 * 1024)) / delta_t
                    ema_speed_mbs = inst_speed_mbs if ema_speed_mbs <= 0 else (ema_alpha * inst_speed_mbs + (1 - ema_alpha) * ema_speed_mbs)
                    last_sample_time = now
                    if total_bytes > 0:
                        pct = min(100.0, (downloaded / total_bytes) * 100.0)
                        bucket = int(pct * 10)
                        should_log = (
                            bucket != last_bucket
                            and pct < 100.0
                            and (
                                (now - last_log_time >= 0.5)
                                if use_inline
                                else (bucket % 50 == 0 or now - last_log_time >= 5.0)
                            )
                        )
                        if should_log:
                            downloaded_mb = downloaded / (1024 * 1024)
                            avg_speed_mbs = downloaded_mb / max(now - start_time, 1e-6)
                            speed_mbs = ema_speed_mbs if ema_speed_mbs > 0 else avg_speed_mbs
                            remaining_mb = max(total_mb - downloaded_mb, 0.0)
                            eta_seconds = remaining_mb / speed_mbs if speed_mbs > 0 else 0.0
                            _emit_progress(
                                f"[QwenVL] {file_label}: {pct:5.1f}% "
                                f"({downloaded_mb:.1f}/{total_mb:.1f} MB) - "
                                f"{speed_mbs:.1f} MB/s - ETA {_fmt_eta(eta_seconds)}"
                            )
                            last_log_time = now
                        if bucket != last_bucket:
                            file_pbar.update_absolute(bucket, 1000, None)
                            last_bucket = bucket

            file_pbar.update_absolute(1000, 1000, None)
            if total_bytes > 0:
                elapsed = max(time.time() - start_time, 1e-6)
                avg_speed_mbs = total_mb / elapsed if elapsed > 0 else 0.0
                final_speed_mbs = ema_speed_mbs if ema_speed_mbs > 0 else avg_speed_mbs
                _emit_progress(
                    f"[QwenVL] {file_label}: 100.0% "
                    f"({total_mb:.1f}/{total_mb:.1f} MB) - {final_speed_mbs:.1f} MB/s - ETA 00:00"
                    ,
                    done=True,
                )
    except Exception as exc:
        if use_inline and last_render_len > 0:
            print()
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        enable_progress_bars()
        print(f"[QwenVL] Streaming download failed for {filename}, fallback to hf_hub_download: {exc}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(target_path.parent),
            local_dir_use_symlinks=False,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.replace(target_path)
        return

    if not tmp_path.exists():
        raise FileNotFoundError(f"[QwenVL] Download temp file missing for {filename}")
    tmp_path.replace(target_path)


class AILab_QwenVL_GGUF_PromptEnhancer:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ENHANCED_OUTPUT",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def __init__(self):
        self.llm = None
        self.current_signature = None
        self.gguf_models = self.load_gguf_models()
        self.styles = STYLES

    @staticmethod
    def load_gguf_models():
        fallback = {
            "base_dir": "LLM/GGUF",
            "models": {},
        }
        if not GGUF_CONFIG_PATH.exists():
            return fallback
        try:
            with open(GGUF_CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except Exception as exc:
            print(f"[QwenVL] gguf_models.json load failed: {exc}")
            return fallback

        base_dir = data.get("base_dir") or fallback["base_dir"]

        models: dict[str, dict] = {}

        # Legacy/custom direct entries (optional)
        legacy_models = data.get("models") or {}
        if isinstance(legacy_models, dict):
            for name, entry in legacy_models.items():
                if isinstance(entry, dict):
                    models[name] = entry

        # Text-only catalog (use Qwen_model; do not use qwenVL_model here)
        qwen_repos = data.get("Qwen_model") or {}
        if isinstance(qwen_repos, dict):
            seen_display_names: set[str] = set()
            for repo_key, repo in qwen_repos.items():
                if not isinstance(repo, dict):
                    continue
                author = repo.get("author") or repo.get("publisher")
                repo_name = repo.get("repo_name") or repo.get("repo_name_override") or repo_key
                defaults = repo.get("defaults") if isinstance(repo.get("defaults"), dict) else {}
                repo_id = repo.get("repo_id")
                alt_repo_ids = repo.get("alt_repo_ids") or []
                model_files = repo.get("model_files") or []
                for model_file in model_files:
                    # Prefer short names in UI: just the filename.
                    display = Path(model_file).name
                    if display in seen_display_names:
                        display = f"{display} ({repo_key})"
                    seen_display_names.add(display)
                    entry = dict(defaults)
                    entry.update(
                        {
                            "author": author,
                            "repo_dirname": repo_name,
                            "repo_id": repo_id,
                            "alt_repo_ids": alt_repo_ids,
                            "filename": model_file,
                        }
                    )
                    models[display] = entry

        return {"base_dir": base_dir, "models": models}

    @classmethod
    def INPUT_TYPES(cls):
        styles = list(STYLES.keys())
        preferred_style = "📝 Enhance"
        default_style = preferred_style if preferred_style in styles else (styles[0] if styles else "📝 Enhance")
        temp = cls.load_gguf_models()
        model_keys = sorted(list((temp.get("models") or {}).keys())) or ["(edit gguf_models.json)"]
        default_model = model_keys[0]
        return {
            "required": {
                "model_name": (model_keys, {"default": default_model, "tooltip": "GGUF model entry defined in gguf_models.json."}),
                "prompt_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt text to enhance. Leave blank to just emit the preset instruction."}),
                "preset_system_prompt": (styles, {"default": default_style}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 256, "min": 32, "max": 1024}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "english_output": ("BOOLEAN", {"default": False, "tooltip": "Force final output in English using translation prompt."}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": "Select device; auto prefers GPU when available."}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
            }
        }

    def clear(self):
        self.llm = None
        self.current_signature = None

    def _resolve_model_path(self, model_name):
        models = self.gguf_models.get("models") or {}
        entry = models.get(model_name) or {}

        # Back-compat: allow workflows to pass a filename instead of a catalog key.
        if not entry:
            wanted = _model_name_to_filename_candidates(model_name)
            for candidate in models.values():
                filename = candidate.get("filename")
                if filename and Path(filename).name in wanted:
                    entry = candidate
                    break

        base_dir = _resolve_base_dir(self.gguf_models.get("base_dir") or "LLM/GGUF")

        path = entry.get("path")
        if path:
            return Path(path).expanduser()

        filename = entry.get("filename")
        if filename:
            author = _safe_dirname(str(entry.get("author") or entry.get("publisher") or ""))
            repo_dir = _safe_dirname(str(entry.get("repo_dirname") or model_name))
            if author and author != "unknown":
                return base_dir / author / repo_dir / Path(filename).name
            return base_dir / repo_dir / Path(filename).name

        return base_dir / model_name

    def _maybe_download_model(self, model_name, resolved):
        if resolved.exists():
            return
        models = self.gguf_models.get("models") or {}
        entry = models.get(model_name) or {}
        if not entry:
            wanted = _model_name_to_filename_candidates(model_name)
            for candidate in models.values():
                filename = candidate.get("filename")
                if filename and Path(filename).name in wanted:
                    entry = candidate
                    break

        repo_ids = [rid for rid in (entry.get("alt_repo_ids") or []) + [entry.get("repo_id")] if rid]
        filename = entry.get("filename") or resolved.name
        if not repo_ids or not filename:
            raise FileNotFoundError(f"[QwenVL] GGUF missing and no repo_id/filename to download: {resolved}")
        target_dir = resolved.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        attempted = []
        total_attempts = max(len(repo_ids), 1)
        pbar = ProgressBar(total_attempts)
        last_render_len = 0

        def _emit_attempt_progress(message: str, done: bool = False):
            nonlocal last_render_len
            pad = " " * max(0, last_render_len - len(message))
            print(f"\r{message}{pad}", end="\n" if done else "", flush=True)
            last_render_len = 0 if done else len(message)

        _emit_attempt_progress(f"[QwenVL] Download attempts: {_format_download_progress(0, total_attempts)}")

        for idx, repo_id in enumerate(repo_ids, start=1):
            attempted.append(repo_id)
            pbar.update_absolute(idx, total_attempts, None)
            print(f"[QwenVL] Downloading GGUF {filename} from {repo_id}")
            try:
                _stream_hf_file_with_percent(repo_id=repo_id, filename=filename, target_path=resolved)
            except Exception as exc:
                print(f"[QwenVL] hf_hub_download failed from {repo_id}: {exc}")
            finally:
                _emit_attempt_progress(f"[QwenVL] Download attempts: {_format_download_progress(idx, total_attempts)}")
            if resolved.exists():
                break
            try:
                enable_progress_bars()
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    allow_patterns=[filename, f"**/{filename}"],
                )
            except Exception as exc:
                print(f"[QwenVL] Filtered snapshot failed from {repo_id}: {exc}")
            if resolved.exists():
                break
            found = list(target_dir.rglob(filename))
            if found:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                found[0].replace(resolved)
                break
        _emit_attempt_progress(f"[QwenVL] Download attempts: {_format_download_progress(total_attempts, total_attempts)}", done=True)
        if not resolved.exists():
            raise FileNotFoundError(f"[QwenVL] GGUF model not found after download: {resolved} (tried: {', '.join(attempted)})")

    def _load_model(self, model_name, device):
        resolved = self._resolve_model_path(model_name)
        self._maybe_download_model(model_name, resolved)
        model_cfg = self.gguf_models["models"].get(model_name, {})
        context_length = model_cfg.get("context_length", 8192)
        signature = (resolved, context_length, device)
        if self.llm is not None and self.current_signature == signature:
            return
        self.clear()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if not resolved.exists():
            raise FileNotFoundError(f"[QwenVL] GGUF model not found: {resolved}")
        print(f"[QwenVL] Loading GGUF model from {resolved}")
        if device == "auto":
            device_choice = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        else:
            device_choice = device
        auto_gpu_layers = -1 if device_choice == "cuda" else 0
        threads = None
        if device_choice == "cpu":
            threads = max(os.cpu_count() or 1, 1)
        kwargs = {
            "model_path": str(resolved),
            "n_ctx": context_length,
            "n_gpu_layers": auto_gpu_layers,
            "n_threads": None if threads == 0 else threads,
            "n_batch": 1024,
            "verbose": False,
            "chat_format": "qwen",
        }
        self.llm = Llama(**kwargs)
        self.current_signature = signature

    def _invoke_llama(
        self,
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        seed,
    ):
        def _looks_like_planning(text: str) -> bool:
            if not text:
                return False
            return bool(
                re.search(
                    r"(?im)^\s*(okay[,.:]?|first[,.:]?|next[,.:]?|then[,.:]?|wait[,.:]?)\b",
                    text,
                )
                or re.search(r"(?i)\b(i\s+(should|need|must|will|am\s+going\s+to|have\s+to))\b", text)
            )

        def _call(system: str, user: str, temp: float, seed_val: int) -> str:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                seed=seed_val,
            )
            if not response or "choices" not in response or not response["choices"]:
                raise RuntimeError("[QwenVL] llama_cpp returned empty response")
            return (response["choices"][0].get("message", {}).get("content", "") or "").strip()

        raw = _call(system_prompt, user_prompt, float(temperature), int(seed))
        cleaned = clean_model_output(raw, OutputCleanConfig(mode="prompt"))

        # If the model only emitted thinking/planning (common with some Qwen variants),
        # do a single constrained retry asking for final prompt text only.
        if not cleaned or _looks_like_planning(cleaned) or "<think" in raw.lower():
            retry_system = (
                "You are a professional photography prompt writer.\n"
                "Output ONLY ONE final photography prompt paragraph.\n"
                "No analysis, no planning steps, no first-person, and no <think>.\n"
                "No bullet points, no headings, no JSON, no markdown, no quotes."
            )
            retry_user = (
                "Rewrite the following into the final prompt paragraph:\n\n"
                f"{raw}\n"
            )
            raw_retry = _call(retry_system, retry_user, 0.4, int(seed) + 999)
            cleaned_retry = clean_model_output(raw_retry, OutputCleanConfig(mode="prompt"))
            if cleaned_retry and not _looks_like_planning(cleaned_retry):
                return cleaned_retry

        return cleaned or ""

    def process(
        self,
        model_name,
        prompt_text,
        preset_system_prompt,
        custom_system_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        english_output,
        device,
        seed,
    ):
        style_entry = self.styles.get(preset_system_prompt, {})
        system_prompt = (custom_system_prompt.strip() or style_entry.get("system_prompt") or "").strip()
        if not system_prompt:
            raise ValueError("system_prompt is empty; check AILab_System_Prompts.json or preset selection.")
        system_prompt = (
            f"{system_prompt}\n\n"
            "Return only the final prompt text. No preface, no explanations, no analysis, no JSON, no markdown fences, and no <think>.\n"
            "Do not write planning steps (no 'First', 'Next', 'Then') and do not use first-person ('I', 'we')."
        )
        user_prompt = prompt_text.strip() or "Describe a scene vividly."
        merged_prompt = user_prompt
        self._load_model(model_name, device)
        enhanced = self._invoke_llama(
            system_prompt=system_prompt,
            user_prompt=merged_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        if english_output:
            translated = self._invoke_llama(
                system_prompt=(
                    PROMPT_CONFIG.get("translation_prompt")
                    or "Return a single English paragraph (150-300 words). No prefixes, bullets, JSON, or <think>. "
                    "Cover subject, environment, lighting, camera settings, composition, color/texture, and style. Output only the prompt."
                ),
                user_prompt=enhanced,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.05,
                seed=seed + 1,
            )
            final = clean_model_output(translated, OutputCleanConfig(mode="prompt")) or translated.strip()
        else:
            final = clean_model_output(enhanced, OutputCleanConfig(mode="prompt")) or enhanced.strip()
        return (final,)

    @staticmethod
    def _is_english(text):
        letters = len(re.findall(r"[A-Za-z]", text))
        tokens = len(re.findall(r"\S", text))
        return tokens > 0 and letters / tokens > 0.7

    @staticmethod
    def _strip_think(text):
        return clean_model_output(text, OutputCleanConfig(mode="prompt"))


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_GGUF_PromptEnhancer": AILab_QwenVL_GGUF_PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_GGUF_PromptEnhancer": "QwenVL Prompt Enhancer (GGUF)",
}
