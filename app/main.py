import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import shutil
from paddleocr import PaddleOCR

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_cudnn_exhaustive_search"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"


app = FastAPI(title="OCR Service")

ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    use_gpu=False
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "OCR API is running"}

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ocr.ocr(file_path, cls=False)

        extracted = []
        full_text = ""

        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                confidence = float(line[1][1])

                extracted.append({
                    "text": text,
                    "confidence": confidence
                })

                full_text += text + " "

        os.remove(file_path)

        return {
            "success": True,
            "text": full_text.strip(),
            "lines": extracted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#uvicorn app.main:app --port 8001