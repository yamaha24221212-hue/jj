import os, io, urllib.parse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from groq import Groq
from gtts import gTTS

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
SYSTEM_PROMPT = ("شما یونا هستید، یک ربات رومیزی کوچک، مهربان و باهوش. "
                 "پاسخ‌ها حداکثر ۲ تا ۳ جمله کوتاه و صمیمی به فارسی باشد.")

@app.get("/")
def home():
    return {"status": "Yona AI Brain is Active!"}

@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    audio = await file.read()
    try:
        tr = client.audio.transcriptions.create(
            file=("input.wav", audio), model="whisper-large-v3", language="fa")
        user_text = tr.text.strip() or "سلام"
        comp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_text}],
            temperature=0.6, max_tokens=150)
        reply = comp.choices[0].message.content.strip()
        return JSONResponse({"user": user_text, "reply": reply})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/tts")
def tts(text: str):
    data = io.BytesIO()
    gTTS(text=text, lang="fa").write_to_fp(data)
    return Response(data.getvalue(), media_type="audio/mpeg")
