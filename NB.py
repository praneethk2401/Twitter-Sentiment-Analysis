import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset into a Pandas dataframe
data = pd.read_csv('modified_dataset.csv')
# Calculate the percentage of positive, negative, and neutral tweets in the dataset
positive_count = (data['Sentiment'] == 'Positive').sum()
negative_count = (data['Sentiment'] == 'Negative').sum()
neutral_count = (data['Sentiment'] == 'Neutral').sum()

total_count = len(data['Sentiment'])

positive_percentage = (positive_count / total_count) * 100
negative_percentage = (negative_count / total_count) * 100
neutral_percentage = (neutral_count / total_count) * 100

# Print the percentage of positive, negative, and neutral tweets in the dataset
print(f"Positive tweets: {positive_percentage:.2f}%")
print(f"Negative tweets: {negative_percentage:.2f}%")
print(f"Neutral tweets: {neutral_percentage:.2f}%")
# Clean the text data by removing stopwords and punctuation
stop_words = set(stopwords.words('english'))
data['cleaned_text'] = data['OriginalTweet'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if (word.isalpha() and word.lower() not in stop_words)]))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['Sentiment'], test_size=0.2, random_state=42)

print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Vectorize the text data using a CountVectorizer
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier on the training set
clf = MultinomialNB()
clf.fit(X_train_vect, y_train)

# Predict the sentiment of the test set using the trained model
y_pred = clf.predict(X_test_vect)

# Calculate accuracy, precision, and recall of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
