from fastapi import FastAPI
from pydantic import BaseModel
import os, sys, traceback
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ===== 환경 =====
BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")  # Render에 환경변수로 넣기
MODEL_NAME = os.getenv("MODEL_NAME", "Junginn/kcelectra-toxic-comment-detector_V1")

# ✅ 토크나이저는 베이스 모델로 분리 (없으면 MODEL_NAME 사용)
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "beomi/KcELECTRA-base-v2022")

token_kw = {"token": HF_AUTH_TOKEN} if HF_AUTH_TOKEN else {}

# 1순위: 베이스 토크나이저(슬로우) 사용
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False, **token_kw)
except Exception:
    # 2순위: 동일 리포에서 fast 시도 (tokenizer.json만 있을 때)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True, **token_kw)

# ===== 모델 로드 =====
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **token_kw)
model.tokenizer = tokenizer
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Render는 보통 CPU
model.to(device)

app = FastAPI()

class PredictIn(BaseModel):
    text: str

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# 디버그: 현재 상태 확인 (토큰/모델명/토크나이저명)
@app.get("/debug/status")
def debug_status():
    return {
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "model_name": MODEL_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "has_token": bool(HF_AUTH_TOKEN),
    }

@app.post("/predict")
def predict(request: PredictIn):
    """
    어떤 예외가 나도 200 JSON으로 돌려서 프론트가 항상 안전하게 처리하도록 설계.
    (필요하면 status_code를 500으로 바꿔도 되지만, 현재 프런트 구조는 200이 편함)
    """
    try:
        text = (request.text or "").strip()
        if not text:
            return {
                "ok": True,
                "text": "",
                "predicted_label": 1,
                "label_name": "일반 댓글",
                "probability": 1.0,
                "confidence_color": None,
            }

        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=150,  # (메모리 타이트하면 128로 낮추세요)
            return_attention_mask=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        label = int(torch.argmax(probs, dim=-1).item())
        prob = float(probs[0][label].item())
        label_text = '악플' if label == 0 else '일반 댓글'

        # 🔴 확신도 기반 색상 정의
        color = None
        if label == 0:  # 악플로 예측된 경우만 색상 부여
            if prob >= 0.65:
                color = "red"
            elif prob >= 0.5:
                color = "orange"
            else:
                # 악플로 예측됐지만 확신 낮음 → 일반으로 간주
                label_text = '일반 댓글'
                # color는 None 유지

        return {
            "ok": True,
            "text": text,
            "predicted_label": label,
            "label_name": label_text,
            "probability": round(prob, 4),
            "confidence_color": color
        }

    except Exception as e:
        # 서버 로그에 원인 남기기
        print("[/predict ERROR]", e, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        # 200으로 JSON 반환(프론트가 파싱 가능)
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "confidence_color": None
        }

# ===== 정적 파일/루트 =====
static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    return (static_dir / "index.html").read_text(encoding="utf-8")
