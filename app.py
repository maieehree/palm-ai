import os
import base64
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

from palm_knowledge import PALM_KNOWLEDGE

load_dotenv(override=True)

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="AI ดูแลปาล์ม",
    description="API วิเคราะห์ภาพทะลายปาล์มน้ำมัน และตอบคำถามผู้เชี่ยวชาญ",
    version="2.0.0",
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
DEFAULT_MODEL = "qwen/qwen3.6-27b"   # รองรับ Vision + Thai ดีมาก (ฟรีบน Groq)
VISION_MAX_TOKENS = 800   # ลดจาก 1500 — วิเคราะห์ภาพไม่ต้องยาว
CHAT_MAX_TOKENS = 600     # ปรับให้พอดี — ไม่ตัดกลางคัน แต่ยังกระชับ

SYSTEM_PROMPT = f"""ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอังกฤษ

คุณคือ "น้องปาล์ม" AI ผู้เชี่ยวชาญปาล์มน้ำมัน สังกัดศูนย์วิจัยปาล์มน้ำมันสุราษฎร์ธานี กรมวิชาการเกษตร

ฐานความรู้:
{PALM_KNOWLEDGE}

กฎตอบคำถาม:
1. ตอบภาษาไทยเท่านั้น
2. ตอบกระชับ ตรงประเด็น ไม่เกิน 5 บรรทัด ยกเว้นวิเคราะห์ภาพ
3. ระบุระดับความสุก สีทะลาย ผลร่วง คำแนะนำ และราคา (บาท/กก.) เมื่อวิเคราะห์ภาพ
4. ถ้าไม่แน่ใจให้บอกตรงๆ อย่าเดา
5. อ้าง มกษ. 5702-2562 เมื่อพูดถึงความสุก
6. ห้ามเกริ่นนำหรือสรุปซ้ำ ตอบตรงๆ เลย
"""


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def get_groq_client() -> Groq:
    load_dotenv(override=True)   # โหลด key ใหม่ทุกครั้ง → เปลี่ยน .env ไม่ต้อง restart
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า GROQ_API_KEY กรุณาใส่ใน .env ก่อน",
        )
    return Groq(api_key=api_key)


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

    # ── Step 2: Line-by-line — บรรทัดที่ไม่มีภาษาไทยเลย → reasoning ──
    THAI_RE = re.compile(r'[\u0e00-\u0e7f]')
    answer_lines = []
    english_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            # บรรทัดว่าง — ใส่ใน answer ถ้า answer มีเนื้อหาแล้ว
            if answer_lines:
                answer_lines.append('')
            elif english_lines:
                english_lines.append('')
            continue

        if THAI_RE.search(stripped):
            # มีอักษรไทย → บรรทัดนี้เป็น answer
            if english_lines:
                # flush English ที่ค้างอยู่ไป reasoning
                reasoning_parts.append('\n'.join(english_lines).strip())
                english_lines = []
            answer_lines.append(line)
        else:
            # ไม่มีอักษรไทยเลย → เป็น English/reasoning
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
    return {"status": "ok", "service": "AI ดูแลปาล์ม", "version": "2.0.0"}


@app.post("/api/analyze")
async def analyze_palm_image(
    file: UploadFile = File(..., description="ไฟล์รูปภาพทะลายปาล์ม"),
    tree_label: str = Form(default="ไม่ระบุ", description="ชื่อต้น/แปลง"),
    zone: str = Form(default="ไม่ระบุ", description="โซนสวน"),
    user_note: str = Form(default="", description="หมายเหตุเพิ่มเติม"),
    model: str = Form(default=DEFAULT_MODEL, description="โมเดล AI ที่ใช้"),
):
    """
    วิเคราะห์ภาพทะลายปาล์มด้วย Qwen Vision AI

    - รับไฟล์รูปภาพ (jpg/png/webp ≤ 20MB)
    - ตรวจสอบระดับความสุก สี ผลร่วง
    - แนะนำวันเก็บเกี่ยวและราคาโดยประมาณ
    """
    # ── ตรวจสอบขนาดไฟล์ (Groq จำกัด 20MB) ──
    image_bytes = await file.read()
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
        response = client.chat.completions.create(
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
            "reasoning": reasoning,   # กระบวนการคิดของ AI ("" ถ้าไม่มี)
            "tree_label": tree_label,
            "zone": zone,
            "model_used": model,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"วิเคราะห์ไม่ได้: {exc}")


@app.post("/api/chat")
async def chat_with_palm_ai(request: ChatRequest):
    """
    คุยกับ AI ผู้เชี่ยวชาญปาล์มน้ำมัน (รองรับ multi-turn)

    - ส่ง messages array ทั้งหมด (history)
    - AI จะตอบโดยอิงฐานความรู้ DOA
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="กรุณาส่งข้อความอย่างน้อย 1 ข้อความ")

    client = get_groq_client()

    # สร้าง system prompt ที่รวมบริบทของเฮังเดอร์ (ถ้ามี)
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
        response = client.chat.completions.create(
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