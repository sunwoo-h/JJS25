from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, torch, sys, traceback
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

# 캐시(선택)
os.environ.setdefault("TRANSFORMERS_CACHE", str(BASE_DIR / ".hf_cache"))
Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)

token_kw = {"token": HF_AUTH_TOKEN} if HF_AUTH_TOKEN else {}

# ===== 토크나이저/모델 로드 =====
# ✅ 핵심: 모델 리포에서 fast 토크나이저 로드 (정확도 유지 포인트)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, **token_kw)

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

# (진단용) 버전 확인
@app.get("/debug/versions")
def versions():
    import transformers, tokenizers
    return {
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "torch": torch.__version__,
        "python": sys.version,
        "model_name": MODEL_NAME,
        "has_token": bool(HF_AUTH_TOKEN),
    }

@app.post("/predict")
def predict(request: PredictIn):
    try:
        text = request.text.strip()
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=150,
            return_attention_mask=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        label = int(torch.argmax(probs, dim=-1).item())
        prob = float(probs[0][label].item())
        label_text = '악플' if label == 0 else '일반 댓글'

        color = None
        if label == 0:  # 악플일 때만 색 부여
            if prob >= 0.65:
                color = "red"
            elif prob >= 0.5:
                color = "orange"
            else:
                label_text = '일반 댓글'  # 확신 낮으면 일반으로 간주

        return {
            "text": text,
            "predicted_label": label,
            "label_name": label_text,
            "probability": round(prob, 4),
            "confidence_color": color
        }
    except Exception as e:
        print("[/predict ERROR]", e, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        # 프론트가 json()을 항상 부를 수 있게 JSON 에러 형태로 반환
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

# ===== 정적 파일/루트 =====
static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    return (static_dir / "index.html").read_text(encoding="utf-8")
