from flask import Flask, render_template, request, session, redirect, url_for
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import timedelta

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")
app.permanent_session_lifetime = timedelta(minutes=60)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt to guide the assistant
SYSTEM_PROMPT = (
    "You are a business discovery bot helping a B2B lead generation platform. "
    "Your goal is to ask 10 intelligent and product-relevant questions to understand the user's offering "
    "so that your company can find the right customers (buyers) for them. "
    "Do not ask questions about industry trends or market predictions. "
    "Do not ask questions the user wouldn't know the answer to. "
    "Focus on product-specific, logistical, and operational information. "
    "You must include these 4 mandatory questions (distributed naturally): "
    "1. Average Turnaround Time 2. Supply Capacity 3. Present Demand 4. Expected Demand."
)

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = [{"role": "system", "content": SYSTEM_PROMPT}]
        session["qa_log"] = []
        session["question_count"] = 0

    if request.method == "POST":
        user_answer = request.form.get("answer")
        current_question = session["history"][-1]["content"]
        session["qa_log"].append({"question": current_question, "answer": user_answer})
        session["history"].append({"role": "user", "content": user_answer})
        session["question_count"] += 1

        if session["question_count"] >= 10:
            return redirect(url_for("complete"))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=session["history"],
            max_tokens=150,
            temperature=0.7
        )

        next_question = response.choices[0].message.content.strip()
        session["history"].append({"role": "assistant", "content": next_question})
        return render_template("index.html", question=next_question, qa_log=session["qa_log"])

    # First GET request — start with the first question
    first_question = "What is your product and what does it do?"
    session["history"].append({"role": "assistant", "content": first_question})
    return render_template("index.html", question=first_question, qa_log=[])


@app.route("/complete")
def complete():
    return render_template("complete.html", qa_log=session.get("qa_log", []))


if __name__ == "__main__":
    app.run(debug=True)
