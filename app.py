# import pickle
# import spacy


# nlp = spacy.load("en_core_web_sm")
# tfid = pickle.load(open("tfid.pkl", "rb"))
# model = pickle.load(open("model.pkl", "rb"))


# def text_transform(text):
#     text = text.lower()
#     text = nlp(text)

#     tokens = []
#     result = []
#     for token in text:
#         if token.is_alpha or token.is_digit:
#             tokens.append(token)

#     tokens = [token for token in tokens if not token.is_punct]
#     tokens = [token for token in tokens if not token.is_stop]

#     for token in tokens:
#         result.append(token.lemma_)

#     result = " ".join(result)
#     return result


# transform_text = text_transform("Valid for 12 hours. Click on this link to WIN $2000 and a trip to Maldives. Use CODE: ILM90 to get 90 percent off!!!")
# transform_text = tfid.transform([transform_text])
# prediction = model.predict(transform_text)
# if prediction[0] == 0:
#     print("It is not a spam email")
# else:
#     print("It is a spam email")


from flask import Flask, request, render_template
import pickle
import spacy

app = Flask(__name__, static_folder='static')
nlp = spacy.load("en_core_web_sm")
tfid = pickle.load(open("tfid.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def text_transform(text):
    text = text.lower()
    text = nlp(text)
    
    tokens = []
    result = []
    for token in text:
        if token.is_alpha or token.is_digit:
            tokens.append(token)
    
    tokens = [token for token in tokens if not token.is_punct]
    tokens = [token for token in tokens if not token.is_stop]

    for token in tokens:
        result.append(token.lemma_)
    
    result = " ".join(result)
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        transformed_text = text_transform(message)
        vectorized_text = tfid.transform([transformed_text])
        prediction = model.predict(vectorized_text)
        
        if prediction[0] == 0:
            result = "Not Spam"
        else:
            result = "Spam"
            
        return render_template('index.html', prediction_text=f'This message is: {result}', message=message)

if __name__ == '__main__':
    app.run(debug=True)