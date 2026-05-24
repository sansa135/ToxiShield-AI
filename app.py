import flask
import sqlite3
import random
import speech_recognition as sr
import matplotlib.pyplot as plt
import pandas as pd

from transformers import pipeline

import google.generativeai as genai

from openai import OpenAI

from reportlab.pdfgen import canvas

# =========================
# FLASK
# =========================

app = flask.Flask(__name__)

app.secret_key = "toxishield"

# =========================
# DATABASE
# =========================

conn = sqlite3.connect(
    "toxishield.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""

               CREATE TABLE IF NOT EXISTS users(

                                                   id INTEGER PRIMARY KEY AUTOINCREMENT,

                                                   username TEXT,

                                                   password TEXT

               )

               """)

cursor.execute("""

               CREATE TABLE IF NOT EXISTS history(

                                                     id INTEGER PRIMARY KEY AUTOINCREMENT,

                                                     comment TEXT,

                                                     result TEXT,

                                                     score REAL,

                                                     emotion TEXT

               )

               """)

conn.commit()

# =========================
# AI MODELS
# =========================

classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

# =========================
# GEMINI API
# =========================

genai.configure(
    api_key=""
)

gemini_model = genai.GenerativeModel(
    "gemini-pro"
)

# =========================
# OPENAI API
# =========================

openai_client = OpenAI(
    api_key=""
)

# =========================
# EMOTION
# =========================

def detect_emotion(score):

    if score > 90:
        return "Extreme Anger 😡"

    elif score > 70:
        return "Aggressive ⚠"

    elif score > 40:
        return "Neutral 🙂"

    else:
        return "Positive 😊"

# =========================
# LOGIN
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if flask.request.method == "POST":

        username = flask.request.form["username"]

        password = flask.request.form["password"]

        cursor.execute(
            """

            SELECT *

            FROM users

            WHERE username=?

              AND password=?

            """,

            (
                username,
                password
            )
        )

        user = cursor.fetchone()

        if user:

            flask.session["user"] = username

            return flask.redirect("/")

    return flask.render_template(
        "login.html"
    )

# =========================
# SIGNUP
# =========================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if flask.request.method == "POST":

        username = flask.request.form["username"]

        password = flask.request.form["password"]

        cursor.execute(
            """

            INSERT INTO users(
                username,
                password
            )

            VALUES(?,?)

            """,

            (
                username,
                password
            )
        )

        conn.commit()

        return flask.redirect("/login")

    return flask.render_template(
        "signup.html"
    )

# =========================
# HOME
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    if "user" not in flask.session:

        return flask.redirect("/login")

    result = ""
    score = 0
    emotion = ""
    rewrite = ""
    color = "green"

    if flask.request.method == "POST":

        comment = flask.request.form["comment"]

        prediction = classifier(comment)

        score = round(
            prediction[0]["score"] * 100,
            2
        )

        emotion = detect_emotion(score)

        if score > 50:

            result = "⚠ Toxic Comment"

            color = "red"

        else:

            result = "✅ Safe Comment"

            color = "lime"

        # GEMINI REWRITE

        gemini_response = gemini_model.generate_content(

            f"Rewrite this politely: {comment}"

        )

        rewrite = gemini_response.text

        # SAVE HISTORY

        cursor.execute(
            """

            INSERT INTO history(
                comment,
                result,
                score,
                emotion
            )

            VALUES(?,?,?,?)

            """,

            (
                comment,
                result,
                score,
                emotion
            )
        )

        conn.commit()

    # ANALYTICS

    cursor.execute(
        "SELECT COUNT(*) FROM history"
    )

    total_scans = cursor.fetchone()[0]

    cursor.execute(
        """

        SELECT COUNT(*)

        FROM history

        WHERE score > 50

        """
    )

    toxic_count = cursor.fetchone()[0]

    safe_count = total_scans - toxic_count

    # PIE CHART

    labels = ["Toxic", "Safe"]

    values = [toxic_count, safe_count]

    plt.figure(figsize=(5,5))

    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    plt.savefig(
        "static/chart.png"
    )

    # HISTORY

    cursor.execute(
        """

        SELECT *

        FROM history

        ORDER BY id DESC

            LIMIT 5

        """
    )

    history = cursor.fetchall()

    return flask.render_template(

        "index.html",

        result=result,

        score=score,

        emotion=emotion,

        rewrite=rewrite,

        color=color,

        history=history,

        total_scans=total_scans,

        toxic_count=toxic_count,

        safe_count=safe_count
    )

# =========================
# PDF REPORT
# =========================

@app.route("/report")
def report():

    pdf = canvas.Canvas(
        "static/report.pdf"
    )

    pdf.drawString(
        100,
        750,
        "ToxiShield AI Report"
    )

    cursor.execute(
        "SELECT * FROM history"
    )

    data = cursor.fetchall()

    y = 700

    for row in data:

        pdf.drawString(
            50,
            y,
            str(row)
        )

        y -= 20

    pdf.save()

    return flask.send_file(
        "static/report.pdf"
    )

# =========================
# VOICE DETECTION
# =========================

@app.route("/voice")
def voice():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        audio = recognizer.listen(source)

    text = recognizer.recognize_google(
        audio
    )

    prediction = classifier(text)

    return {

        "voice_text": text,

        "prediction": prediction
    }


# =========================
# ADMIN DASHBOARD
# =========================

@app.route("/admin")
def admin():

    cursor.execute(
        "SELECT * FROM users"
    )

    users = cursor.fetchall()

    cursor.execute(
        "SELECT * FROM history"
    )

    history = cursor.fetchall()

    return flask.render_template(
        "admin.html",
        users=users,
        history=history
    )

# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(debug=True)