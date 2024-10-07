import pickle
import spacy


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


transform_text = text_transform()
transform_text = tfid.transform([transform_text])
prediction = model.predict(transform_text)
if prediction[0] == 0:
    print("It is not a spam email")
else:
    print("It is a spam email")
