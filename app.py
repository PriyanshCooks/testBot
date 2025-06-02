import os
import uuid
import subprocess
from flask import Flask, render_template, request, session, redirect, url_for
from openai import OpenAI
from dotenv import load_dotenv
from datetime import timedelta
from pydantic import BaseModel, Field
from typing import List, Literal
import json

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")
app.permanent_session_lifetime = timedelta(minutes=60)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = "chat_sessions"  # folder to store JSON files

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class QAItem(BaseModel):
    role: Literal["assistant", "user"]
    question: str = ""
    answer: str = ""

class Conversation(BaseModel):
    conversation: List[QAItem] = Field(default_factory=list)

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

def ask_openai(prompt, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def get_conversation_from_file(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return Conversation.parse_obj(data)
    except Exception:
        return Conversation()

def save_conversation_to_file(conversation: Conversation, filename):
    with open(filename, "w") as f:
        f.write(conversation.model_dump_json(indent=2))

def git_commit_and_push_with_token(filename):
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("GitHub token not found.")
        return

    remote_url = f"https://{github_token}@github.com/PriyanshCooks/testBot.git"

    try:
        # Add remote only if not already added
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
    except subprocess.CalledProcessError:
        # If remote already exists, update it
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)

    try:
        subprocess.run(["git", "add", filename], check=True)
        subprocess.run(["git", "commit", "-m", f"Add/update {os.path.basename(filename)}"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Git Error] {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_file" not in session:
        session["chat_file"] = os.path.join(DATA_DIR, f"chat_{uuid.uuid4().hex}.json")

    filename = session["chat_file"]
    conversation = get_conversation_from_file(filename)

    history = []
    for item in conversation.conversation:
        if item.role == "assistant":
            history.append({"role": "assistant", "content": item.question})
        elif item.role == "user":
            history.append({"role": "user", "content": item.answer})

    if request.method == "POST":
        user_answer = request.form.get("answer")
        if not user_answer:
            last_question = conversation.conversation[-1].question if conversation.conversation else "What is your product and what does it do?"
            return render_template("index.html", question=last_question, qa_log=conversation.conversation)

        last_question = conversation.conversation[-1].question if conversation.conversation else "What is your product and what does it do?"
        conversation.conversation.append(QAItem(role="user", question="", answer=user_answer))
        history.append({"role": "user", "content": user_answer})

        if len([q for q in conversation.conversation if q.role == "assistant"]) >= 10:
            save_conversation_to_file(conversation, filename)
            git_commit_and_push_with_token(filename)  # 🔁 Push to GitHub
            return redirect(url_for("complete"))

        next_question = ask_openai(
            "Based on the previous Q&A, ask the next most relevant question strictly related to understanding"
            " the user’s product, its logistics, buyer requirements, and supply-readiness."
            " You must cover all 4 of these before the 10th question if not already covered:"
            " Turnaround Time, Supply Capacity, Present Demand, Expected Demand."
            " Do NOT ask about market trends or insights. Do NOT ask for user’s analysis of the market."
            " Ask only what the user would realistically know and what helps find customers."
            " Avoid redundancy.",
            history
        )

        conversation.conversation.append(QAItem(role="assistant", question=next_question, answer=""))
        save_conversation_to_file(conversation, filename)

        return render_template("index.html", question=next_question, qa_log=conversation.conversation)

    conversation = Conversation()
    first_question = "What is your product and what does it do?"
    conversation.conversation.append(QAItem(role="assistant", question=first_question, answer=""))
    save_conversation_to_file(conversation, filename)

    return render_template("index.html", question=first_question, qa_log=conversation.conversation)

@app.route("/complete")
def complete():
    filename = session.get("chat_file")
    if not filename or not os.path.exists(filename):
        return "No conversation found.", 404

    conversation = get_conversation_from_file(filename)
    return render_template("complete.html", qa_log=conversation.conversation)

if __name__ == "__main__":
    app.run(debug=True)
