from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# البيانات
texts = ["I love AI", "This is great", "I hate bugs", "This is terrible"]
labels = ["positive", "positive", "negative", "negative"]

# تحويل النصوص إلى شعاع
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# تدريب النموذج
model = MultinomialNB()
model.fit(X, labels)

# تجربة
test_text = ["AI is awesome", "bugs are annoying"]
test_vec = vectorizer.transform(test_text)
predictions = model.predict(test_vec)

# نتائج
for text, label in zip(test_text, predictions):
    print(f"Text: {text} => Prediction: {label}")