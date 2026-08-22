import os
import base64
import time
import random
from typing import List, Optional
from pathlib import Path

import sys
VENV_SITE_PACKAGES = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq


class RateLimitError(Exception):
    """Raised when Groq API rate limit is reached after retries."""


from palm_knowledge import PALM_KNOWLEDGE

load_dotenv(override=True)

# ─────────────────────────────────────────────
# Feature flag: Tiny Vision Pipeline
# ─────────────────────────────────────────────
# USE_TINY_VISION=true  → Image → Classifier → JSON → Groq NLP (no vision)
# USE_TINY_VISION=false → Image → Groq Vision (pipeline เดิม, default)
#
# เปิด-ปิดได้ใน .env โดยไม่ต้องแก้ code
# ──────────────────────────────────────────────
USE_TINY_VISION: bool = os.getenv("USE_TINY_VISION", "false").lower() == "true"

# Emergency mode: force classifier-only operation (no Groq calls)
# Can be set via .env or toggled at runtime via /admin/emergency
EMERGENCY_CLASSIFIER_ONLY = os.getenv("EMERGENCY_CLASSIFIER_ONLY", "false").lower() == "true"

# If true, when classifier reports low confidence, return a best-effort classifier
# estimate (avoid blocking UI) instead of returning success=false. Useful when
# Groq is unavailable or to restore responsiveness. Set in .env if desired.
ALLOW_LOW_CONFIDENCE_FALLBACK = os.getenv("ALLOW_LOW_CONFIDENCE_FALLBACK", "true").lower() in ("1","true","yes","on")

# Vision module (lazy import — ไม่ crash ถ้า USE_TINY_VISION=false)
_vision_available = False
if USE_TINY_VISION:
    try:
        from backend.vision.preprocessing import validate_and_load_image, ImageValidationError
        from backend.vision.classifier import classify_image, RIPENESS_LABELS
        _vision_available = True
        print(f"[usekedo] ✅ Tiny Vision pipeline ENABLED (rule-based classifier)")
    except ImportError as _e:
        print(f"[usekedo] ⚠️  USE_TINY_VISION=true but vision module not available: {_e}")
        print("[usekedo]    Falling back to OLD pipeline.")
        USE_TINY_VISION = False
else:
    print("[usekedo] ℹ️  USE_TINY_VISION=false — using original Qwen Vision pipeline")

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="usekedo",
    description="API วิเคราะห์ภาพทะลายปาล์มน้ำมัน และตอบคำถามผู้เชี่ยวชาญ",
    version="2.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULT_MODEL   = "llama-3.1-8b-instant"  # เร็วมาก 1-3 วิ บน Groq
VISION_MAX_TOKENS = 400  # ให้วิเคราะห์ภาพได้ละเอียดพอ
CHAT_MAX_TOKENS   = 300  # ตอบภาษาไทยได้เต็มที่

# ── Retry config for Groq 429 ──
MAX_RETRIES = 2          # retry น้อยลง ไม่ให้ user รอนาน
BASE_WAIT   = 0.5        # backoff เร็วขึ้น
MAX_WAIT    = 8.0        # รอสูงสุดไม่เกิน 8 วิ

# Default full prompt (used when not in tiny-vision mode)
FULL_SYSTEM_PROMPT = f"""ตอบภาษาไทยเท่านั้น กระชับ ตรงประเด็น
คุณคือ usekedo ผู้เชี่ยวชาญปาล์มน้ำมัน

{PALM_KNOWLEDGE}

กฎ: ตอบสั้น ไม่เกิน 5 บรรทัด ระบุความสุก สีทะลาย ราคา(บาท/กก.) ห้ามเกริ่นนำ ตอบตรงๆ
"""

# Short prompt used when tiny vision pipeline is enabled — instruct model to
# return only the ripeness label in Thai (single token) to minimize latency.
SHORT_SYSTEM_PROMPT = (
    "ตอบเป็นภาษาไทยคำเดียวเท่านั้น: ระบุระดับความสุกของทะลายปาล์ม หนึ่งในตัวเลือกต่อไปนี้ "
    "'ยังไม่สุก', 'ใกล้สุก', 'สุกพอดี', หรือ 'สุกเกิน'" 
)

# Select system prompt based on pipeline mode
SYSTEM_PROMPT = SHORT_SYSTEM_PROMPT if USE_TINY_VISION else FULL_SYSTEM_PROMPT

# ─────────────────────────────────────────────
# Intent Guard — หัวข้อที่ usekedo ตอบได้
# ─────────────────────────────────────────────
PALM_KEYWORDS = {
    # ปาล์มและเกษตร
    "ปาล์ม", "ทะลาย", "ผลปาล์ม", "น้ำมันปาล์ม", "สวนปาล์ม", "ต้นปาล์ม",
    "ตัด", "เก็บเกี่ยว", "สุก", "ราคา", "แปลง", "โซน", "ปุ๋ย", "โรค",
    "แมลง", "ศัตรูพืช", "รด", "น้ำ", "ดิน", "ปลูก", "เกษตร", "สวน",
    "ผล", "กก", "บาท", "มกษ", "doa", "กรม", "วิชาการ",
    # คำทั่วไปในบริบทสวนปาล์ม
    "วิเคราะห์", "ภาพ", "รูป", "สี", "ความสุก", "คำแนะนำ", "เปอร์เซ็นต์",
    "น้ำมัน", "ทะลาย", "ช่วง", "ฤดู", "อากาศ", "ฝน", "แล้ง",
    "ต้น", "ใบ", "ราก", "ลำต้น", "ทะลาย", "ผลิ",
}

OFF_TOPIC_REPLY = (
    "ขออภัยครับ usekedo ตอบได้เฉพาะเรื่องปาล์มน้ำมัน การเก็บเกี่ยว "
    "ราคา และการดูแลสวนเท่านั้นครับ หากมีคำถามเรื่องสวนปาล์ม "
    "ยินดีช่วยเสมอครับ 🌴"
)

def is_palm_related(text: str) -> bool:
    """
    Intent guard: คืน True ถ้าข้อความเกี่ยวกับปาล์ม/เกษตร
    ใช้ keyword matching แบบง่าย + fallback ให้ผ่านถ้าไม่แน่ใจ
    (ดีกว่า false negative ที่ block คำถามดีๆ)
    """
    lower = text.lower()
    for kw in PALM_KEYWORDS:
        if kw in lower:
            return True

    # ถ้าสั้นมาก (≤ 10 ตัวอักษร) → ให้ผ่านเพื่อ UX
    if len(text.strip()) <= 10:
        return True

    # keyword ภาษาอังกฤษที่เกี่ยวข้อง
    en_keywords = {"palm", "oil", "harvest", "ripeness", "fruit", "crop",
                   "farm", "plantation", "fertilizer", "price", "ton"}
    for kw in en_keywords:
        if kw in lower:
            return True

    return False


# ─────────────────────────────────────────────
# Helper: Groq client + Retry with backoff
# ─────────────────────────────────────────────
def get_groq_client() -> Groq:
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า GROQ_API_KEY กรุณาใส่ใน .env ก่อน",
        )
    # Reuse a single Groq client instance to avoid repeated initialization overhead
    global _GROQ_CLIENT
    try:
        _GROQ_CLIENT
    except NameError:
        _GROQ_CLIENT = None

    # Track the API key used to create the client so we can detect changes
    global _GROQ_API_KEY_USED
    try:
        _GROQ_API_KEY_USED
    except NameError:
        _GROQ_API_KEY_USED = None

    # If there is no client yet, or the env key changed, (re)create client
    if _GROQ_CLIENT is None or (_GROQ_API_KEY_USED and _GROQ_API_KEY_USED != api_key):
        # Avoid printing or logging the raw key
        if _GROQ_CLIENT is None:
            print("[Groq] Initializing Groq client")
        else:
            print("[Groq] GROQ_API_KEY changed — recreating Groq client to pick up new key")

        _GROQ_CLIENT = Groq(api_key=api_key)
        _GROQ_API_KEY_USED = api_key

    return _GROQ_CLIENT


def groq_call_with_retry(client: Groq, **kwargs):
    """
    เรียก client.chat.completions.create() พร้อม exponential backoff
    รองรับ Groq HTTP 429 (rate limit) และ 503 (overload)
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            err_str = str(exc).lower()
            # First, detect if this error is a Tokens-Per-Day (TPD) / daily quota exhaustion.
            # Example messages observed in logs: "Rate limit reached for model ... on tokens per day (TPD): Limit 200000, Used 197133, Requested 5395. Please try again in 18h12m."
            import re as _re

            is_tpd = False
            # explicit phrases
            if "tokens per day" in err_str or "tokens_per_day" in err_str or "tpd" in err_str:
                is_tpd = True

            # pattern like 'in 18h12m' or '18h' indicates waiting hours (daily quota)
            if not is_tpd:
                m = _re.search(r"(\b(\d{1,3})h(\d{1,2})m\b)", err_str)
                if m:
                    try:
                        hours = int(m.group(2))
                        # any multi-hour wait strongly suggests daily quota exhaustion
                        if hours >= 1:
                            is_tpd = True
                    except Exception:
                        pass

            # If it's clearly a TPD (daily tokens exhausted), do NOT retry — fail fast so caller can fallback.
            if is_tpd:
                print(f"[Groq][TPD] Detected daily token quota exhaustion — skipping retries: {exc}")
                msg = (
                    "Groq API tokens-per-day quota exhausted (TPD). "
                    "Not retrying to avoid long waits; fallback should be used."
                )
                raise RateLimitError(msg)

            # Otherwise, check for retryable transient errors (429, short rate-limit, 503, overload)
            is_retryable = (
                "429" in err_str
                or "rate_limit" in err_str
                or "rate limit" in err_str
                or "503" in err_str
                or "overload" in err_str
                or "server error" in err_str
            )
            if not is_retryable:
                raise   # ถ้าไม่ใช่ rate limit/transient → raise ทันที

            last_exc = exc
            wait = min(BASE_WAIT * (2 ** attempt) + random.uniform(0, 0.5), MAX_WAIT)
            print(f"[Groq] transient error attempt {attempt+1}/{MAX_RETRIES} — wait {wait:.1f}s | {exc}")
            time.sleep(wait)

    # หมด retry แล้วยังไม่ได้ → แจ้งเป็น RateLimitError ให้ caller ตัดสินใจ fallback
    msg = (
        f"Groq API ยุ่งมาก (rate limit) กรุณารอสักครู่แล้วลองใหม่ครับ "
        f"(retry {MAX_RETRIES} ครั้งแล้ว — {last_exc})"
    )
    raise RateLimitError(msg)


# Admin endpoint to toggle emergency classifier-only mode at runtime
@app.post("/admin/emergency")
def set_emergency_mode(on: Optional[bool] = Form(None)):
    """Toggle emergency classifier-only mode.

    POST form data: `on=1` or `on=true` to enable, `on=0`/`false` to disable.
    This changes an in-memory flag (not persisted to .env). Use for urgent mitigation.
    """
    global EMERGENCY_CLASSIFIER_ONLY
    if on is None:
        # return current status
        return {"emergency": EMERGENCY_CLASSIFIER_ONLY}

    val = str(on).lower() in ("1", "true", "yes", "on")
    EMERGENCY_CLASSIFIER_ONLY = val
    return {"emergency": EMERGENCY_CLASSIFIER_ONLY}


# ─────────────────────────────────────────────
# Other helpers
# ─────────────────────────────────────────────
def parse_ripeness(text: str) -> str:
    """ดึงระดับความสุกจาก response ของ AI"""
    if "สุกเกิน" in text:
        return "สุกเกิน"
    if "สุกพอดี" in text:
        return "สุกพอดี"
    if "ใกล้สุก" in text:
        return "ใกล้สุก"
    if "ยังไม่สุก" in text or "ไม่สุก" in text:
        return "ยังไม่สุก"
    return "ไม่ทราบ"


import re

def extract_thinking(text: str) -> tuple:
    """
    แยก Thinking/Reasoning ของ Qwen3 ออกจาก final answer

    วิธีการ:
    1. ดึง <think>...</think> block → reasoning
    2. แยก line-by-line: บรรทัดที่ไม่มีอักษรไทย → reasoning, บรรทัดที่มีไทย → answer
       (วิธีนี้ robust กว่า pattern matching เพราะไม่ขึ้นกับ pattern ที่โมเดลใช้)

    Return: (answer, reasoning) tuple
    """
    if not text:
        return text, ""

    reasoning_parts = []

    # ── Step 1: ดึง <think>...</think> block ──
    think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    if think_match:
        reasoning_parts.append(think_match.group(1).strip())
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # ── Step 2: Line-by-line — บรรทัดที่ "ขึ้นต้นด้วยไทย" → answer, อื่น → reasoning ──
    THAI_RE = re.compile(r'[\u0e00-\u0e7f]')
    BULLET_PREFIX = re.compile(r'^[\s\-\*\•\d\.\)\(：:]+')

    def is_thai_lead(line: str) -> bool:
        """ตรวจว่าบรรทัดขึ้นต้นด้วยอักษรไทย (หลังตัด bullet/เลข prefix)"""
        cleaned = BULLET_PREFIX.sub('', line.strip()).strip()
        if not cleaned:
            return False
        return '\u0e00' <= cleaned[0] <= '\u0e7f'

    answer_lines = []
    english_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if answer_lines:
                answer_lines.append('')
            elif english_lines:
                english_lines.append('')
            continue

        if is_thai_lead(stripped):
            # ขึ้นต้นด้วยไทย → answer
            if english_lines:
                reasoning_parts.append('\n'.join(english_lines).strip())
                english_lines = []
            answer_lines.append(line)
        else:
            # ขึ้นต้นด้วย English หรือสัญลักษณ์ → reasoning
            english_lines.append(line)

    # flush English ที่เหลือ
    if english_lines:
        reasoning_parts.append('\n'.join(english_lines).strip())

    answer = '\n'.join(answer_lines).strip()
    reasoning = '\n\n'.join(r for r in reasoning_parts if r).strip()

    # ── Fallback: ถ้า answer ว่างเปล่า ให้ใช้ text เดิมทั้งหมด ──
    if not answer:
        answer = text.strip()

    return answer, reasoning


RIPENESS_COLOR = {
    "สุกพอดี":  "#C0392B",
    "ใกล้สุก":  "#E8862E",
    "ยังไม่สุก": "#E2B33C",
    "สุกเกิน":  "#6C3483",
    "ไม่ทราบ":  "#7F8C8D",
}


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = DEFAULT_MODEL
    analysis_context: str = ""  # ผลการวิเคราะห์ภาพล่าสุด (ถ้ามี) จะใส่ที่ท้าย system prompt


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """ตรวจสอบสถานะ server"""
    return {
        "status": "ok",
        "service": "usekedo",
        "version": "2.3.0",
        "tiny_vision_enabled": USE_TINY_VISION,
        "tiny_vision_method": "rule_based" if (USE_TINY_VISION and _vision_available) else None,
    }


@app.post("/admin/warmup")
async def admin_warmup(run_groq: bool = False):
    """Warm up classifier (and optionally Groq client) to reduce cold-start latency.

    POST /admin/warmup with JSON body {"run_groq": true} will also instantiate Groq client
    (does not send model requests unless explicitly implemented).
    """
    from PIL import Image, ImageDraw

    # Create a small synthetic palm-like image and non-palm image
    palm = Image.new("RGB", (128, 128), (34, 139, 34))
    d = ImageDraw.Draw(palm)
    d.ellipse((36, 36, 92, 92), fill=(220, 100, 20))

    nonp = Image.new("RGB", (128, 128), (30, 144, 255))

    # Run classifier on both images (populate cache)
    try:
        classify_image(palm)
        classify_image(nonp)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    # Optionally instantiate Groq client (no calls)
    if run_groq:
        try:
            _ = get_groq_client()
        except Exception as exc:
            return {"ok": False, "error": f"groq warmup failed: {exc}"}

    return {"ok": True, "msg": "warmup completed"}


@app.on_event("startup")
async def _on_startup():
    """Automatic warm-up on startup to prime classifier cache and optionally Groq client.

    Controlled by env var `AUTO_WARMUP` (default true) and `WARMUP_RUN_GROQ` (default false).
    """
    aw = os.getenv("AUTO_WARMUP", "true").lower() in ("1", "true", "yes")
    run_groq = os.getenv("WARMUP_RUN_GROQ", "false").lower() in ("1", "true", "yes")
    if not aw:
        return

    try:
        # call warmup routine
        await admin_warmup(run_groq=run_groq)
    except Exception:
        # do not crash startup on warmup failure
        pass


@app.post("/api/analyze")
async def analyze_palm_image(
    file: UploadFile = File(..., description="ไฟล์รูปภาพทะลายปาล์ม"),
    tree_label: str = Form(default="ไม่ระบุ", description="ชื่อต้น/แปลง"),
    zone: str = Form(default="ไม่ระบุ", description="โซนสวน"),
    user_note: str = Form(default="", description="หมายเหตุเพิ่มเติม"),
    model: str = Form(default=DEFAULT_MODEL, description="โมเดล AI ที่ใช้"),
):
    """
    วิเคราะห์ภาพทะลายปาล์ม

    Pipeline ถูกเลือกตาม USE_TINY_VISION environment variable:
      - USE_TINY_VISION=false (default): Image → Qwen Vision AI (pipeline เดิม)
      - USE_TINY_VISION=true:  Image → Tiny Classifier → JSON → Qwen NLP

    รับไฟล์รูปภาพ (jpg/png/webp)
    ขีดจำกัดขนาด: MAX_IMAGE_SIZE_MB (default 10MB), สูงสุด 20MB สำหรับ pipeline เดิม
    """
    # ── อ่านไฟล์ ──
    image_bytes = await file.read()

    # ════════════════════════════════════════════════════════
    # NEW PIPELINE: USE_TINY_VISION=true
    # Image → Validate → Classify → JSON → Groq NLP
    # (ภาพไม่ถูกส่งไป Groq ในกรณีนี้)
    # ════════════════════════════════════════════════════════
    if USE_TINY_VISION and _vision_available:
        return await _analyze_tiny_vision(
            image_bytes=image_bytes,
            filename=file.filename or "upload",
            tree_label=tree_label,
            zone=zone,
            user_note=user_note,
            model=model,
        )

    # ════════════════════════════════════════════════════════
    # OLD PIPELINE: USE_TINY_VISION=false (original, unchanged)
    # Image → Groq Vision → Answer
    # ════════════════════════════════════════════════════════
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="ไฟล์ภาพใหญ่เกิน 20MB กรุณาลดขนาดก่อน")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    content_type = file.content_type or "image/jpeg"

    client = get_groq_client()

    vision_prompt = f"""ต้นที่/แปลง: {tree_label} | โซน: {zone}{f' | หมายเหตุ: {user_note}' if user_note else ''}

วิเคราะห์ทะลายปาล์มในภาพ ตอบเป็นภาษาไทย กระชับ ครบ 4 หัวข้อนี้:
1. ระดับความสุก (ระบุ 1 ใน: ยังไม่สุก / ใกล้สุก / สุกพอดี / สุกเกิน)
2. สีและลักษณะทะลาย + ผลร่วงโดยประมาณ
3. เปอร์เซ็นต์น้ำมันโดยประมาณ
4. คำแนะนำ: ควรเก็บเมื่อไหร่ + ราคาโดยประมาณ (บาท/กก.)
ห้ามเกริ่นนำ ตอบตรงๆ เลย"""

    try:
        response = groq_call_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": vision_prompt},
                    ],
                },
            ],
            temperature=0.3,
            max_tokens=VISION_MAX_TOKENS,
        )

        analysis_text, reasoning = extract_thinking(response.choices[0].message.content)
        ripeness = parse_ripeness(analysis_text)

        return {
            "success": True,
            "ripeness": ripeness,
            "ripeness_color": RIPENESS_COLOR.get(ripeness, "#7F8C8D"),
            "full_analysis": analysis_text,
            "reasoning": reasoning,
            "tree_label": tree_label,
            "zone": zone,
            "model_used": model,
            "pipeline": "old_vision",   # บอก client ว่าใช้ pipeline ไหน
        }
    except RateLimitError as rl:
        # Groq rate limited — try a best-effort classifier fallback if available
        if USE_TINY_VISION and _vision_available:
            try:
                from backend.vision.preprocessing import validate_and_load_image, ImageValidationError
                from backend.vision.classifier import classify_image

                img = validate_and_load_image(image_bytes, filename=file.filename or "upload")
                clf = classify_image(img)
                ripeness_display = RIPENESS_LABELS.get(clf.ripeness or "", "ไม่ทราบ")
                return {
                    "success": True,
                    "ripeness": clf.ripeness,
                    "ripeness_color": RIPENESS_COLOR.get(ripeness_display, "#7F8C8D"),
                    "full_analysis": f"(fallback) ระบุโดย classifier: {ripeness_display}",
                    "reasoning": "(fallback) Groq API rate-limited; returning classifier estimate",
                    "tree_label": tree_label,
                    "zone": zone,
                    "model_used": "tiny_classifier",
                    "pipeline": "fallback_classifier",
                }
            except Exception:
                raise HTTPException(status_code=503, detail=f"Groq rate limit and classifier fallback failed: {rl}")

        # No classifier available — return 503 with friendly message
        raise HTTPException(status_code=503, detail=f"Groq API rate limit (retry later): {rl}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"วิเคราะห์ไม่ได้: {exc}")


# ─────────────────────────────────────────────
# NEW PIPELINE HANDLER (internal)
# ─────────────────────────────────────────────
async def _analyze_tiny_vision(
    *,
    image_bytes: bytes,
    filename: str,
    tree_label: str,
    zone: str,
    user_note: str,
    model: str,
) -> dict:
    """
    NEW pipeline handler:
      1. Validate image (MIME, size, magic bytes)
      2. Classify with Tiny Image Classifier
      3. If not_palm → return error immediately (NO Groq call)
      4. If low_confidence → return warning (NO Groq call)
      5. Build structured JSON → send to Groq NLP (no image)
      6. Return analysis result

    Groq never receives a raw image in this pipeline.
    """
    import json as _json

    # ── Step 1: Validate ────────────────────────────────────────────
    try:
        img = validate_and_load_image(image_bytes, filename=filename)
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    # ── Step 2: Classify ────────────────────────────────────────────
    try:
        clf_result = classify_image(img)
    except Exception as exc:
        # If classifier crashes, fall back gracefully to old pipeline
        raise HTTPException(
            status_code=500,
            detail=f"Classifier error: {exc}"
        )

    # FAST PATH: If classifier is confident, return concise structured result
    # without calling the LLM. This reduces latency dramatically.
    # Criteria: classifier reports is_palm True, not low_confidence, and
    # ripeness_confidence high enough.
    try:
        fast_conf_threshold = float(os.getenv("FAST_PATH_CONFIDENCE", "0.75"))
    except Exception:
        fast_conf_threshold = 0.75

    if clf_result.is_palm and not clf_result.low_confidence and (clf_result.ripeness_confidence or 0) >= fast_conf_threshold:
        # Simple heuristics to provide quick numeric estimates so LLM is not needed
        ripeness_map_oil = {
            "unripe": (3, 7),
            "nearly_ripe": (9, 13),
            "ripe": (14, 20),
            "overripe": (10, 14),
        }
        low, high = ripeness_map_oil.get(clf_result.ripeness or "ripe", (8, 15))
        est_oil_pct = round((low + high) / 2, 1)

        # Price heuristic: base price per kg adjusted by ripeness quality
        base_price = float(os.getenv("BASE_PALM_PRICE", "6.5"))  # บาท/กก.
        ripeness_price_adj = {
            "unripe": 0.9,
            "nearly_ripe": 1.0,
            "ripe": 1.1,
            "overripe": 0.95,
        }
        price = round(base_price * ripeness_price_adj.get(clf_result.ripeness or "ripe", 1.0), 2)

        return {
            "success": True,
            "fast_path": True,
            "is_palm": True,
            "palm_confidence": clf_result.palm_confidence,
            "ripeness": clf_result.ripeness,
            "ripeness_display": RIPENESS_LABELS.get(clf_result.ripeness or "", "ไม่ทราบ"),
            "ripeness_confidence": clf_result.ripeness_confidence,
            "estimated_oil_percent": est_oil_pct,
            "estimated_price_baht_per_kg": price,
            "note": "Classifier confident — returned quick estimate without LLM",
            "model_used": "tiny_classifier",
            "pipeline": "new_tiny_vision_fast",
            "classifier_method": clf_result.method,
        }

    # Emergency mode: force classifier-only behavior (never call Groq)
    if EMERGENCY_CLASSIFIER_ONLY:
        # Provide the best classifier-only estimate we can and avoid calling Groq
        ripeness_map_oil = {
            "unripe": (3, 7),
            "nearly_ripe": (9, 13),
            "ripe": (14, 20),
            "overripe": (10, 14),
        }
        low, high = ripeness_map_oil.get(clf_result.ripeness or "ripe", (8, 15))
        est_oil_pct = round((low + high) / 2, 1)
        base_price = float(os.getenv("BASE_PALM_PRICE", "6.5"))
        ripeness_price_adj = {
            "unripe": 0.9,
            "nearly_ripe": 1.0,
            "ripe": 1.1,
            "overripe": 0.95,
        }
        price = round(base_price * ripeness_price_adj.get(clf_result.ripeness or "ripe", 1.0), 2)

        return {
            "success": True,
            "fast_path": True,
            "is_palm": bool(clf_result.is_palm),
            "palm_confidence": clf_result.palm_confidence,
            "ripeness": clf_result.ripeness,
            "ripeness_display": RIPENESS_LABELS.get(clf_result.ripeness or "", "ไม่ทราบ"),
            "ripeness_confidence": clf_result.ripeness_confidence,
            "estimated_oil_percent": est_oil_pct,
            "estimated_price_baht_per_kg": price,
            "note": "Emergency mode: returning classifier-only estimate (no LLM)",
            "model_used": "tiny_classifier",
            "pipeline": "new_tiny_vision_emergency",
            "classifier_method": clf_result.method,
        }

    # ── Step 3: Not-a-palm guard ─────────────────────────────────────
    # Security: ภาพที่ถูก reject ต้องไม่ถูกส่งไป Groq
    if not clf_result.is_palm:
        return {
            "success": False,
            "is_palm": False,
            "palm_confidence": clf_result.palm_confidence,
            "ripeness": None,
            "ripeness_color": "#7F8C8D",
            "full_analysis": (
                "ภาพนี้ไม่ใช่ภาพปาล์มน้ำมัน กรุณาอัปโหลดภาพทะลายปาล์มที่ชัดเจน"
            ),
            "reasoning": "",
            "tree_label": tree_label,
            "zone": zone,
            "model_used": "tiny_classifier",
            "pipeline": "new_tiny_vision",
            "classifier_method": clf_result.method,
        }

    # ── Step 4: Low-confidence guard ─────────────────────────────────
    if clf_result.low_confidence:
        ripeness_display = RIPENESS_LABELS.get(clf_result.ripeness or "", "ไม่ทราบ")
        # If fallback enabled, return best-effort classifier estimate instead of blocking
        if ALLOW_LOW_CONFIDENCE_FALLBACK:
            ripeness_map_oil = {
                "unripe": (3, 7),
                "nearly_ripe": (9, 13),
                "ripe": (14, 20),
                "overripe": (10, 14),
            }
            low, high = ripeness_map_oil.get(clf_result.ripeness or "ripe", (8, 15))
            est_oil_pct = round((low + high) / 2, 1)
            base_price = float(os.getenv("BASE_PALM_PRICE", "6.5"))
            ripeness_price_adj = {
                "unripe": 0.9,
                "nearly_ripe": 1.0,
                "ripe": 1.1,
                "overripe": 0.95,
            }
            price = round(base_price * ripeness_price_adj.get(clf_result.ripeness or "ripe", 1.0), 2)

            return {
                "success": True,
                "fast_path": False,
                "is_palm": True,
                "palm_confidence": clf_result.palm_confidence,
                "ripeness": clf_result.ripeness,
                "ripeness_display": ripeness_display,
                "ripeness_confidence": clf_result.ripeness_confidence,
                "estimated_oil_percent": est_oil_pct,
                "estimated_price_baht_per_kg": price,
                "note": "Classifier low confidence — returning best-effort estimate",
                "model_used": "tiny_classifier",
                "pipeline": "new_tiny_vision_lowconf_fallback",
                "classifier_method": clf_result.method,
                "low_confidence": True,
            }

        # Default behavior: inform user and request better photo
        return {
            "success": False,
            "is_palm": True,
            "palm_confidence": clf_result.palm_confidence,
            "ripeness": clf_result.ripeness,
            "ripeness_color": RIPENESS_COLOR.get(ripeness_display, "#7F8C8D"),
            "full_analysis": (
                "ไม่สามารถระบุระดับความสุกได้อย่างมั่นใจ "
                "กรุณาอัปโหลดภาพที่เห็นทะลายชัดขึ้น "
                "(แสงธรรมชาติ ภาพไม่เบลอ เห็นสีผลชัดเจน)"
            ),
            "reasoning": "",
            "tree_label": tree_label,
            "zone": zone,
            "model_used": "tiny_classifier",
            "pipeline": "new_tiny_vision",
            "classifier_method": clf_result.method,
            "low_confidence": True,
        }

    # ── Step 5: Build structured JSON for Groq NLP ───────────────────
    ripeness_display = RIPENESS_LABELS.get(clf_result.ripeness or "", "ไม่ทราบ")

    vision_json_payload = {
        "is_palm":              clf_result.is_palm,
        "ripeness":             clf_result.ripeness,
        "ripeness_display":     ripeness_display,
        "ripeness_confidence":  clf_result.ripeness_confidence,
        "palm_confidence":      clf_result.palm_confidence,
        "classifier_method":    clf_result.method,
    }

    user_context = f"ต้นที่/แปลง: {tree_label} | โซน: {zone}"
    if user_note:
        user_context += f" | หมายเหตุ: {user_note}"

    nlp_prompt = (
        f"{user_context}\n\n"
        f"ผลการตรวจจับภาพทะลายปาล์มจาก Image Classifier:\n"
        f"{_json.dumps(vision_json_payload, ensure_ascii=False, indent=2)}\n\n"
        "จากผลการวิเคราะห์ข้างต้น กรุณาตอบเป็นภาษาไทย กระชับ ครบ 4 หัวข้อ:\n"
        "1. ระดับความสุก (ยืนยันจากผลที่ได้)\n"
        "2. สีและลักษณะทะลายที่คาดว่าจะเห็น\n"
        "3. เปอร์เซ็นต์น้ำมันโดยประมาณ\n"
        "4. คำแนะนำ: ควรเก็บเมื่อไหร่ + ราคาโดยประมาณ (บาท/กก.)\n"
        "ห้ามเกริ่นนำ ตอบตรงๆ เลย"
    )

    # ── Step 6: Groq NLP (text only — no image) ──────────────────────
    client = get_groq_client()
    try:
        response = groq_call_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": nlp_prompt},
            ],
            temperature=0.3,
            max_tokens=VISION_MAX_TOKENS,
        )

        analysis_text, reasoning = extract_thinking(response.choices[0].message.content)
        ripeness = parse_ripeness(analysis_text) or ripeness_display

        return {
            "success": True,
            "is_palm": True,
            "palm_confidence": clf_result.palm_confidence,
            "ripeness": ripeness,
            "ripeness_color": RIPENESS_COLOR.get(ripeness, "#7F8C8D"),
            "full_analysis": analysis_text,
            "reasoning": reasoning,
            "tree_label": tree_label,
            "zone": zone,
            "model_used": model,
            "pipeline": "new_tiny_vision",
            "classifier_method": clf_result.method,
            "classifier_ripeness": clf_result.ripeness,
            "classifier_confidence": clf_result.ripeness_confidence,
        }
    except RateLimitError as rl:
        # Groq rate limit — return classifier-only estimate as fallback
        return {
            "success": True,
            "is_palm": True,
            "palm_confidence": clf_result.palm_confidence,
            "ripeness": clf_result.ripeness,
            "ripeness_color": RIPENESS_COLOR.get(ripeness_display, "#7F8C8D"),
            "full_analysis": f"(fallback) ระบุโดย classifier: {ripeness_display}",
            "reasoning": "(fallback) Groq API rate-limited; returning classifier estimate",
            "tree_label": tree_label,
            "zone": zone,
            "model_used": "tiny_classifier",
            "pipeline": "new_tiny_vision_fallback",
            "classifier_method": clf_result.method,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"วิเคราะห์ไม่ได้: {exc}")


@app.post("/api/chat")
async def chat_with_palm_ai(request: ChatRequest):
    """
    คุยกับ usekedo ผู้เชี่ยวชาญปาล์มน้ำมัน (รองรับ multi-turn)

    - ส่ง messages array ทั้งหมด (history)
    - AI จะตอบโดยอิงฐานความรู้ DOA
    - มี intent guard กรองคำถามนอกเรื่อง
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="กรุณาส่งข้อความอย่างน้อย 1 ข้อความ")

    # ── Intent Guard: กรองคำถามนอกเรื่องปาล์ม ──
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )
    if last_user_msg and not is_palm_related(last_user_msg):
        return {
            "success": True,
            "response": OFF_TOPIC_REPLY,
            "reasoning": "",
            "model_used": request.model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "intent_blocked": True,   # flag ให้ frontend รู้ว่าถูก guard block
        }

    client = get_groq_client()
    # Emergency: if emergency classifier-only mode is enabled, skip Groq and reply friendly
    if EMERGENCY_CLASSIFIER_ONLY:
        return {
            "success": False,
            "response": "ขณะนี้บริการ LLM ถูกปิดชั่วคราว (โหมดฉุกเฉิน) — กรุณาลองใหม่ทีหลัง หรือใช้ผลการวิเคราะห์จากภาพที่แสดงอยู่",
            "reasoning": "EMERGENCY_CLASSIFIER_ONLY enabled",
            "model_used": request.model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "rate_limited": True,
        }

    # สร้าง system prompt ที่รวมบริบทการวิเคราะห์ (ถ้ามี)
    system_content = SYSTEM_PROMPT
    if request.analysis_context:
        system_content += f"""

[ผลการวิเคราะห์ภาพล่าสุดที่เกษตรกรส่งมา]
{request.analysis_context}
[สิ้นสุดผลการวิเคราะห์]
ใช้ข้อมูลนี้ในการตอบคำถามต่อไปด้วย"""

    # Build message list with system prompt at top
    messages: list = [{"role": "system", "content": system_content}]
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail=f"role ไม่ถูกต้อง: {msg.role}")
        messages.append({"role": msg.role, "content": msg.content})

    try:
        response = groq_call_with_retry(
            client,
            model=request.model,
            messages=messages,
            temperature=0.4,
            max_tokens=CHAT_MAX_TOKENS,
        )

        answer, reasoning = extract_thinking(response.choices[0].message.content)

        return {
            "success": True,
            "response": answer,
            "reasoning": reasoning,   # กระบวนการคิดของ AI ("" ถ้าไม่มี)
            "model_used": request.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }

    except RateLimitError as rl:
        # Inform client that LLM is temporarily unavailable
        return {
            "success": False,
            "response": "ขออภัย ขณะนี้ระบบ LLM ยุ่งมาก กรุณาลองใหม่ภายหลัง (Rate limit)",
            "reasoning": str(rl),
            "model_used": request.model,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "rate_limited": True,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ตอบไม่ได้: {exc}")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)