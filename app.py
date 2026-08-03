from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import faqs

app = Flask(__name__)

questions = list(faqs.keys())
answers = list(faqs.values())

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_best_answer(user_question):
    user_vector = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score < 0.2:
        return "Sorry, I couldn't find a relevant answer. Please ask another AI-related question."

    return answers[best_match]

@app.route("/", methods=["GET", "POST"])
def home():
    bot_response = ""

    if request.method == "POST":
        user_question = request.form["question"]
        bot_response = get_best_answer(user_question)

    return render_template("index.html", response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)