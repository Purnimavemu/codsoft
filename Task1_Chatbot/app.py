from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def chatbot_response(user_input):
    user_input = user_input.lower()

    if user_input in ["hi", "hello"]:
        return "Hello! How can I help you?"

    elif "how are you" in user_input:
        return "I'm doing great! Thanks for asking."

    elif "your name" in user_input:
        return "I am a simple AI chatbot."

    elif "what can you do" in user_input:
        return "I can answer simple questions."

    elif "bye" in user_input:
        return "Goodbye! Have a nice day."

    else:
        return "Sorry, I don't understand."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot():
    user_message = request.form["msg"]
    response = chatbot_response(user_message)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
