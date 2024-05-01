import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import spacy

# Sample dataset containing user utterances and their intents
data = {
    'utterance': ['What is the weather like today?', 'Book a flight to New York', 'Set an alarm for 7am', 'Buy milk'],
    'intent': ['weather', 'book_flight', 'set_alarm', 'buy']
}
df = pd.DataFrame(data)

# Intent Recognition
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['utterance'])
y_intent = df['intent']
intent_classifier = MultinomialNB()
intent_classifier.fit(X, y_intent)

# Entity Extraction using SpaCy's Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Dialog Management
class DialogManager:
    def __init__(self, intent_classifier):
        self.intent_classifier = intent_classifier

    def process_input(self, text):
        intent = self.intent_classifier.predict(tfidf_vectorizer.transform([text]))[0]
        entities = extract_entities(text)
        response = self.generate_response(intent, entities)
        return response

    def generate_response(self, intent, entities):
        if intent == 'weather':
            return 'The weather is sunny today.'
        elif intent == 'book_flight':
            destination = entities[0][0] if entities else 'your destination'
            return f'Booking a flight to {destination}.'
        elif intent == 'set_alarm':
            time = entities[0][0] if entities else '7am'
            return f'Alarm set for {time}.'
        elif intent == 'buy':
            item = entities[0][0] if entities else 'milk'
            return f'Buying {item}.'

# Example usage
dialog_manager = DialogManager(intent_classifier)
user_input = "What's the weather like tomorrow?"
response = dialog_manager.process_input(user_input)
print("User:", user_input)
print("Chatbot:", response)
