from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_msg = db.Column(db.String(500))
    bot_reply = db.Column(db.String(500))

with app.app_context():
    db.create_all()

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = model["vectorizer"]
X = model["x"]              # ✔ Correct key from your model
answers = model["answers"]
questions = model["question"]   # ✔ Correct key from your model

def chatbot_response(user_text):
    user_vec = vectorizer.transform([user_text])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    return answers[index]

# Routes
@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data["message"]
    bot_reply = chatbot_response(user_msg)

    # Save chat to DB
    chat_entry = ChatHistory(user_msg=user_msg, bot_reply=bot_reply)
    db.session.add(chat_entry)
    db.session.commit()

    return jsonify({"response": bot_reply})

@app.route("/history/<int:limit>")
def history(limit):
    chats = ChatHistory.query.limit(limit).all()
    output = [{"user": c.user_msg, "bot": c.bot_reply} for c in chats]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
