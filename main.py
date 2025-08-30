# main.py (전체 교체)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ===== 기본 환경 =====
BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")  # Render 대시보드에 Secret으로 넣기
MODEL_NAME = os.getenv("MODEL_NAME", "Junginn/kcelectra-toxic-comment-detector_V1")

# ✅ 토크나이저는 베이스 모델로 분리 (없으면 MODEL_NAME 사용)
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "beomi/KcELECTRA-base-v2022")

# (선택) 캐시 경로 고정
os.environ.setdefault("TRANSFORMERS_CACHE", str(BASE_DIR / ".hf_cache"))
Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)

# (선택) 병렬/스레드 최소화 — 메모리/안정성에 도움
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

token_kw = {"token": HF_AUTH_TOKEN} if HF_AUTH_TOKEN else {}

app = FastAPI()

# ========== (A) 지연 로딩 & tokenizer 연결 ==========
_tokenizer = None
_model = None
_device = torch.device("cpu")  # Render는 보통 CPU

def load_model_if_needed():
    """최초 요청 시에만 토크나이저/모델을 메모리에 올림."""
    global _tokenizer, _model

    if _tokenizer is None:
        # 슬로우 토크나이저 우선 (vocab.txt 기반)
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_NAME, use_fast=False, **token_kw
            )
        except Exception:
            # 필요 시 Fast 토크나이저로 폴백
            _tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_NAME, use_fast=True, **token_kw
            )

    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True,  # 로딩 피크 메모리 절감
            **token_kw,
        )
        _model.to(_device)
        _model.eval()
        # ✅ 중요: 모델 객체에 토크나이저를 붙여둔다(예전 코드 호환)
        _model.tokenizer = _tokenizer

# ========= 데이터 모델 =========
class PredictIn(BaseModel):
    text: str

# ========= 헬스 체크 =========
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# ========== (B) 예외를 JSON으로 반환하는 /predict ==========
@app.post("/predict")
def predict(req: PredictIn):
    try:
        load_model_if_needed()  # 첫 호출 때만 메모리 로딩
        text = req.text.strip()

        # _model.tokenizer 대신 _tokenizer 직접 사용해도 OK
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=150,  # 필요하면 128로 낮춰도 됨
            return_attention_mask=True,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        label = torch.argmax(probs, dim=-1).item()
        prob = probs[0][label].item()
        label_text = "악플" if label == 0 else "일반 댓글"

        color = None
        if label == 0:
            if prob >= 0.65:
                color = "red"
            elif prob >= 0.5:
                color = "orange"
            else:
                label_text = "일반 댓글"

        return {
            "ok": True,
            "text": text,
            "predicted_label": label,
            "label_name": label_text,
            "probability": round(prob, 4),
            "confidence_color": color,
        }

    except Exception as e:
        # 서버 로그로 원인 추적
        print("[/predict ERROR]", e, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        # 항상 JSON으로 반환되게 처리(프론트가 res.json() 해도 안전)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# ===== 정적 파일/루트 =====
static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    return (static_dir / "index.html").read_text(encoding="utf-8")
