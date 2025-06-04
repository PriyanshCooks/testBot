import os
import uuid
from flask import Flask, render_template, request, session, redirect, url_for
from openai import OpenAI
from dotenv import load_dotenv
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
import pymysql
from fuzzywuzzy import fuzz


pymysql.install_as_MySQLdb()  # Make pymysql a drop-in replacement for MySQLdb

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")
app.permanent_session_lifetime = timedelta(minutes=60)

# MySQL connection info from environment variables (matching your .env)
mysql_user = os.getenv("DATABASE_USERNAME")
mysql_password = os.getenv("DATABASE_PASSWORD")
mysql_host = os.getenv("DATABASE_HOST")
mysql_db = os.getenv("DATABASE_NAME")
mysql_port = os.getenv("MYSQL_PORT", 3306)  # optional, default 3306

# Construct MySQL URI for SQLAlchemy using PyMySQL driver
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Models for DB
class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_uuid = db.Column(db.String(64), unique=True, nullable=False)
    qa_items = db.relationship('QAItem', backref='chat_session', lazy=True)

class QAItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # "assistant" or "user"
    question = db.Column(db.Text, default="")
    answer = db.Column(db.Text, default="")

SYSTEM_PROMPT = """
You are a product discovery assistant tasked with collecting essential factual information about a client’s product.

Your task is to collect essential information about a client's product in a professional and conversational tone.

Only ask one question at a time and wait for the user's reply before asking the next one.

You are suppose to collect information so that you can find customers for the client, so ask accordingly but remember these rules.
Important rules:
1. ONLY ask questions from the APPROVED list below. Do not invent new ones.
2. After each answer, use it to decide which question to ask next. Maintain context.
3. Do not ask redundant or repetitive questions.
4. Never ask about things a normal client won't know (like market size, future demand, or trends).
5. Stop asking questions when you feel enough info is collected (max 10 questions usually).
6. Rephrase questions slightly if needed, but keep their intent unchanged.
7. NEVER ask more than one question at a time.
8. ONLY ask about the current customers or the market client is Selling too.
10. DO ask anything which is out of client's Knowledge

Approved Questions List:
1. What is the name or model of the product?
2. What does this product do, and what problem does it solve?
3. What industries or use-cases does this product serve?
4. What are the key features or technical specifications?
5. What is your current production capacity (per month/year)?
6. What is the minimum order quantity (MOQ)?
7. Are there specific regions or countries you are ready to supply to?
8. Can you provide private labeling or custom packaging if required?
9. Who are your current or typical customers (industries, business types)?
10. Are you open to distributors?
11. Which geographic regions are you currently supplying to?
12. Are there any certifications the product complies with?
13. What makes your product better or different from competitors?
14. What feedback do you usually get from repeat clients?
15. Have you supplied this product for any notable projects or brands?
16. What are your after-sales services?
17. Are you currently looking to enter new markets or industries?
18. Is there any additional information that would help us position your product to the right clients?

You must never ask a question that is not directly adapted from this list.

Begin by greeting the client and asking the most basic question to identify the product.
"""

forbidden_phrases = [
    "expected demand", 
    "future demand", 
    "market forecast", 
    "how much future demand",
    "how much demand", 
    "estimate future sales",
    "foresee any increase in demand",
    "market size",
    "current market size",
    "future market size"
]

def is_forbidden(question):
    return any(phrase in question.lower() for phrase in forbidden_phrases)

def is_duplicate(question, qa_items, threshold=80):
    for item in qa_items:
        if item.role == "assistant":
            similarity = fuzz.ratio(item.question.lower(), question.lower())
            if similarity >= threshold:
                return True
    return False


def ask_openai(prompt, history, qa_items):
    def generate_response(messages, temperature=0.7):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    # Build initial messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    # First attempt
    question = generate_response(messages, temperature=0.7)

    if is_forbidden(question) or is_duplicate(question, qa_items):
        # Retry once with a stricter system instruction appended
        retry_system_prompt = SYSTEM_PROMPT + (
            "\nAvoid forbidden topics like demand forecasting or vague future trends. "
            "Do not repeat previously asked questions. Ask only useful, new questions."
        )
        retry_messages = [{"role": "system", "content": retry_system_prompt}]
        retry_messages.extend(history)

        # Retry with lower temperature for less randomness
        question = generate_response(retry_messages, temperature=0.3)

        if is_forbidden(question) or is_duplicate(question, qa_items):
            question = "Thank you. That’s all the questions we needed for now."  # fallback

    return question


def get_chat_session(session_uuid):
    return ChatSession.query.filter_by(session_uuid=session_uuid).first()

def create_chat_session():
    session_uuid = str(uuid.uuid4())
    new_session = ChatSession(session_uuid=session_uuid)
    db.session.add(new_session)
    db.session.commit()
    return new_session

def get_qa_history(chat_session):
    return QAItem.query.filter_by(chat_session_id=chat_session.id).order_by(QAItem.id).all()

def build_history(qa_items):
    history = []
    for item in qa_items:
        if item.role == "assistant":
            history.append({"role": "assistant", "content": item.question})
        elif item.role == "user":
            history.append({"role": "user", "content": item.answer})
    return history


@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_uuid" not in session:
        chat_session = create_chat_session()
        session["chat_uuid"] = chat_session.session_uuid
    else:
        chat_session = get_chat_session(session["chat_uuid"])
        if not chat_session:
            chat_session = create_chat_session()
            session["chat_uuid"] = chat_session.session_uuid

    qa_items = get_qa_history(chat_session)
    history = build_history(qa_items)

    if request.method == "POST":
        user_answer = request.form.get("answer")
        if not user_answer or user_answer.strip() == "":
            last_question = None
            assistant_items = [item for item in qa_items if item.role == "assistant"]
            user_items = [item for item in qa_items if item.role == "user"]
            if len(user_items) < len(assistant_items):
                last_question = assistant_items[len(user_items)].question
            else:
                last_question = "What is your product and what does it do?"
            return render_template("index.html", question=last_question, qa_log=qa_items)

        user_qa = QAItem(chat_session_id=chat_session.id, role="user", question="", answer=user_answer)
        db.session.add(user_qa)
        db.session.commit()
        qa_items = get_qa_history(chat_session)

        history.append({"role": "user", "content": user_answer})

        assistant_questions_count = QAItem.query.filter_by(chat_session_id=chat_session.id, role="assistant").count()
        if assistant_questions_count >= 10:
            return redirect(url_for("complete"))

        next_question = ask_openai(
            "Based on the previous Q&A, ask the next most relevant question strictly related to understanding"
            " the user’s product, its logistics, buyer requirements, and supply-readiness."
            " You must cover all 4 of these before the 10th question if not already covered:"
            " Turnaround Time, Supply Capacity, Present Demand, Expected Demand."
            " Do NOT ask about market trends or insights. Do NOT ask for user’s analysis of the market."
            " Ask only what the user would realistically know and what helps find customers."
            " Avoid redundancy.",
            history,
            qa_items,
        )

        assistant_qa = QAItem(chat_session_id=chat_session.id, role="assistant", question=next_question, answer="")
        db.session.add(assistant_qa)
        db.session.commit()
        qa_items = get_qa_history(chat_session)
        return render_template("index.html", question=next_question, qa_log=qa_items)

    if not qa_items:
        first_question = "What is your product and what does it do?"
        assistant_qa = QAItem(chat_session_id=chat_session.id, role="assistant", question=first_question, answer="")
        db.session.add(assistant_qa)
        db.session.commit()
        qa_items = get_qa_history(chat_session)
    else:
        assistant_items = [item for item in qa_items if item.role == "assistant"]
        user_items = [item for item in qa_items if item.role == "user"]
        if len(user_items) < len(assistant_items):
            first_question = assistant_items[len(user_items)].question
        else:
            first_question = assistant_items[-1].question if assistant_items else "What is your product and what does it do?"

    return render_template("index.html", question=first_question, qa_log=qa_items)

@app.route("/complete")
def complete():
    chat_uuid = session.get("chat_uuid")
    if not chat_uuid:
        return "No conversation found.", 404

    chat_session = get_chat_session(chat_uuid)
    if not chat_session:
        return "No conversation found.", 404

    qa_items = get_qa_history(chat_session)
    return render_template("complete.html", qa_log=qa_items)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if not exist
    app.run(debug=True)
