import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Beispiel-Tickets
data = {
    "text": [
        "I cannot log into my account",
        "My password is not working",
        "I forgot my password",
        "Login failed again",

        "The VPN connection is not working",
        "Internet is very slow today",
        "I cannot connect to the company network",
        "WiFi disconnects all the time",

        "I received a suspicious email",
        "Possible phishing message in my inbox",
        "Someone tried to access my account",
        "I think my account was hacked",

        "The application crashes on startup",
        "Outlook is not opening",
        "The software freezes when I click save",
        "Printer software shows an error"
    ],
    "label": [
        "Password Issue",
        "Password Issue",
        "Password Issue",
        "Password Issue",

        "Network Issue",
        "Network Issue",
        "Network Issue",
        "Network Issue",

        "Security Incident",
        "Security Incident",
        "Security Incident",
        "Security Incident",

        "Software Issue",
        "Software Issue",
        "Software Issue",
        "Software Issue"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"].str.lower())
y = df["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

user_ticket = input("Enter a helpdesk/security ticket: ")
user_vector = vectorizer.transform([user_ticket.lower()])

prediction = model.predict(user_vector)

print(f"Predicted category: {prediction[0]}")