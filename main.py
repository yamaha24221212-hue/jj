import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from groq import Groq
from gtts import gTTS
import io

app = FastAPI()

# خواندن کلید از متغیر محیطی
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# پیام سیستمی برای تعیین شخصیت یونا
SYSTEM_PROMPT = (
    "شما یونا (Yona) هستید، یک ربات رومیزی کوچک، مهربان، باهوش و دوست‌داشتنی برای صبا و جواد. "
    "همیشه کوتاه، صمیمی، دقیق و به زبان شیرین فارسی پاسخ بده. پاسخ‌هایت حداکثر ۲ تا ۳ جمله کوتاه باشد."
)

@app.get("/")
def home():
    return {"status": "Yona AI Brain is Active!"}

@app.post("/chat")
async def chat_audio(file: UploadFile = File(...)):
    # 1. دریافت فایل صوتی ارسالی از میکروفون ESP32
    audio_bytes = await file.read()
    
    # 2. تبدیل گفتار به متن با Whisper در Groq
    transcription = client.audio.transcriptions.create(
        file=("input.wav", audio_bytes),
        model="whisper-large-v3",
        language="fa"
    )
    user_text = transcription.text.strip()
    
    if not user_text:
        user_text = "سلام"

    # 3. پردازش متن با مدل فوق سریع Llama 3 در Groq
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        temperature=0.6,
        max_tokens=150
    )
    bot_reply = completion.choices[0].message.content.strip()

    # 4. تبدیل متن پاسخ به صدای فارسی با gTTS
    tts = gTTS(text=bot_reply, lang="fa", slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # 5. بازگرداندن فایل صوتی برای پخش توسط اسپیکر ربات
    return Response(
        content=mp3_fp.read(),
        media_type="audio/mpeg",
        headers={"X-User-Text": user_text, "X-Bot-Reply": bot_reply}
    )
