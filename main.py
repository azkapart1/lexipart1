import os
import logging
import asyncio
import io
import re
from datetime import datetime, timedelta
import requests
import google.generativeai as genai
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (ApplicationBuilder, CommandHandler, MessageHandler,
                          ContextTypes, filters, CallbackQueryHandler)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.lib import colors
from PyPDF2 import PdfReader, PdfWriter

# === Load API Keys ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PRODUCT_SECRET_KEY = os.getenv("PRODUCT_SECRET_KEY")

# === Configure Gemini ===
genai.configure(api_key=GEMINI_API_KEY)

user_essay_data = {}
user_license_status = {}
logging.basicConfig(level=logging.INFO)


# === AI Feedback Generation ===
def generate_feedback_sync(essay: str) -> str:
    prompt = f"""
You are an IELTS examiner. Evaluate the following essay using the 4 IELTS writing criteria:
- Task Achievement
- Vocabulary
- Grammatical Range & Accuracy
- Coherence & Cohesion

Return the band score and one-sentence comment for each component on a separate line, like:
Task Achievement: 7 - Good understanding but lacks detail.
Vocabulary: 8 - Rich vocabulary with only a few inaccuracies.
Grammatical Range & Accuracy: 7 - Some errors affect clarity.
Coherence & Cohesion: 8 - Well structured with logical flow.
Remember that the band score should be a whole number not half number for each component. Give some mistake examples in the comment.

Then give a brief overall impression summarizing the band scores and comments. Note : Do not include the band prediction sentence and avoid AI biased assessment.

Essay:
{essay}
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error from Gemini: {e}")
        return "❌ Sorry, something went wrong while analyzing your essay."


def extract_band_details(text: str):
    criteria = {
        "Task Achievement":
        r"task achievement[:\-\s]*([\d\.]+)\s*[-–]\s*(.*?)\n",
        "Vocabulary":
        r"vocabulary[:\-\s]*([\d\.]+)\s*[-–]\s*(.*?)\n",
        "Grammatical Range & Accuracy":
        r"grammatical range(?: and| &)? accuracy[:\-\s]*([\d\.]+)\s*[-–]\s*(.*?)\n",
        "Coherence & Cohesion":
        r"coherence(?: and| &)? cohesion[:\-\s]*([\d\.]+)\s*[-–]\s*(.*?)\n"
    }
    summary = "📊 *Band Score Breakdown:*\n\n"
    comments = {}
    scores = []
    for label, pattern in criteria.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            band = float(match.group(1))
            comment = match.group(2).strip()
            scores.append(band)
            comments[label] = (band, comment)
            summary += f"*{label}*: {band}\n💬 {comment}\n\n"
    if scores:
        overall = round(sum(scores) / len(scores) * 2) / 2
        summary += f"*Overall Band Score*: {overall}"
    else:
        summary += "_Band scores not found._"
    return summary.strip(), comments


def extract_overall_comment(text: str) -> str:
    lines = text.splitlines()
    for line in lines:
        if re.match(r"(?i)^overall", line.strip()):
            cleaned = re.sub(r"(?i)^overall(?: impression)?[:\-\s]*", "",
                             line).strip()
            cleaned = re.sub(r"This essay would likely score.*?(?:\\.|$)",
                             "",
                             cleaned,
                             flags=re.IGNORECASE).strip()
            if len(cleaned.split()) >= 5:
                return cleaned

    fallback = re.search(
        r"coherence(?: and| &)? cohesion[:\-\s]*[\d\.]+.*?\n\n(.*)", text,
        re.IGNORECASE | re.DOTALL)
    if fallback:
        paragraph = fallback.group(1).strip().split("\n")[0]
        cleaned = re.sub(r"This essay would likely score.*?(?:\\.|$)",
                         "",
                         paragraph,
                         flags=re.IGNORECASE).strip()
        return cleaned

    return "_No overall comment found._"


used_licenses = {}  # In-memory store


def check_license_validity(user_id, key):
    url = f"https://payhip.com/api/v2/license/verify?license_key={key}"
    headers = {"product-secret-key": PRODUCT_SECRET_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"[Payhip] Status code: {response.status_code}")
            return False, None

        data = response.json()
        print(f"[Payhip] Response JSON: {data}")

        # If already used by a different user
        if key in used_licenses and used_licenses[key] != user_id:
            print(f"[Payhip] License already used by another user.")
            return False, None

        # Accept only if the key has 0 uses and not bound to another user
        if "data" in data and data["data"].get("uses", -1) == 0:
            expiry = datetime.now() + timedelta(days=30)
            user_license_status[user_id] = {"expiry": expiry}
            used_licenses[key] = user_id  # Bind license to user
            return True, expiry
        else:
            print(f"[Payhip] License is invalid or already used.")
            return False, None

    except Exception as e:
        print(f"[Payhip] Exception: {e}")
        return False, None


# === PDF generation ===
def generate_pdf_with_template(comments_dict: dict,
                               overall: str) -> io.BytesIO:
    buffer = io.BytesIO()
    width, height = A4

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    c.setFont("Helvetica", 10)

    y = height - (height * 0.20)
    box_height = 70
    box_width = width - 100

    for label, (score, comment) in comments_dict.items():
        c.setStrokeColor(colors.grey)
        c.setFillColor(colors.whitesmoke)
        c.rect(50, y - box_height, box_width, box_height, fill=1)

        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, y - 20, label)
        c.drawRightString(width - 60, y - 20, f"Band: {score}")

        c.setFont("Helvetica", 9)
        comment_lines = simpleSplit(comment, "Helvetica", 9, box_width - 20)
        text = c.beginText(60, y - 35)
        for line in comment_lines[:3]:
            text.textLine(line)
        c.drawText(text)

        y -= box_height + 20

    overall_score = round(
        sum(score for score, _ in comments_dict.values()) /
        len(comments_dict) * 2) / 2
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y - 10, f"Overall Band Score: {overall_score}")
    c.setFont("Helvetica", 9)
    overall_lines = simpleSplit(overall, "Helvetica", 9, box_width)
    text = c.beginText(50, y - 30)
    for line in overall_lines[:5]:
        text.textLine(line)
    c.drawText(text)

    c.save()
    packet.seek(0)

    overlay_pdf = PdfReader(packet)
    template_pdf = PdfReader("PDF.pdf")
    writer = PdfWriter()
    page = template_pdf.pages[0]
    page.merge_page(overlay_pdf.pages[0])
    writer.add_page(page)

    final = io.BytesIO()
    writer.write(final)
    final.seek(0)
    return final


# === Telegram Bot Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Send your IELTS essay and I’ll analyze it and return: band scores, comments, and optionally a PDF report.\nLimit: 3 free essays."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *How to Use the Bot:*\n\n"
        "1. Send your IELTS essay text.\n"
        "2. Receive feedback, band scores, and word count.\n"
        "3. Tap 📄 to receive a detailed PDF report.\n\n"
        "You get 3 essays for free. After that, please purchase a license key.\n\n"
        "🔑 Already purchased? Use /redeem <key>\n"
        "🛍️ Buy here: https://payhip.com/b/IGJcD\n\n"
        "💬 *Need help?* Contact @LexiBand",
        parse_mode="Markdown")


async def handle_essay(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    essay = update.message.text.strip()
    word_count = len(essay.split())
    count = user_essay_data.get(user_id, {}).get("count", 0)

    # Check if licensed user
    expiry = user_license_status.get(user_id, {}).get("expiry")
    is_licensed = expiry and expiry > datetime.datetime.now()

    if count >= 3 and not is_licensed:
        await update.message.reply_text(
            "🚫 You’ve reached your free limit. Buy a license key here: https://payhip.com/b/IGJcD"
        )
        return

    await update.message.reply_text(
        f"✍️ Analyzing your essay... (Words: {word_count})")
    feedback = await asyncio.to_thread(generate_feedback_sync, essay)
    summary, comments = extract_band_details(feedback)
    overall = extract_overall_comment(feedback)

    user_essay_data[user_id] = {
        "count": count + 1,
        "feedback": feedback,
        "summary": summary,
        "comments": comments,
        "overall": overall
    }

    keyboard = [[
        InlineKeyboardButton("📄 Create PDF Report",
                             callback_data="generate_pdf")
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(f"{summary}\n\n📝 *Overall*: {overall}",
                                    parse_mode="Markdown",
                                    reply_markup=reply_markup)


async def handle_pdf_request(update: Update,
                             context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if user_id not in user_essay_data:
        await query.edit_message_text("❌ No essay data found.")
        return

    data = user_essay_data[user_id]
    pdf_file = generate_pdf_with_template(data["comments"], data["overall"])
    await query.message.reply_document(
        document=InputFile(pdf_file, filename="IELTS_Feedback.pdf"))


async def handle_redeem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    parts = update.message.text.strip().split()
    if len(parts) != 2:
        await update.message.reply_text("❌ Usage: /redeem <LICENSE_KEY>")
        return

    key = parts[1]
    valid, expiry = check_license_validity(user_id, key)

    if valid:
        await update.message.reply_text(
            f"✅ License activated! Expires on {expiry.strftime('%Y-%m-%d')}")
    else:
        await update.message.reply_text(
            "❌ Invalid or already-used license key.")


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    status = user_license_status.get(user_id)
    if status:
        expiry = status["expiry"].strftime('%Y-%m-%d')
        await update.message.reply_text(
            f"🔓 Your license is active until {expiry}.")
    else:
        await update.message.reply_text(
            "🔒 You are using the free version. Use /redeem to activate a license."
        )


# === Entrypoint ===
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("redeem", handle_redeem))
    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_essay))
    app.add_handler(CallbackQueryHandler(handle_pdf_request))
    print("✅ Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
