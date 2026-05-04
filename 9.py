import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd

nltk.download("punkt")  # Для токенізації
nltk.download("punkt_tab")  # Для токенізації речень
nltk.download("stopwords")  # Стоп-слова
nltk.download("wordnet")  # Для лемматизації
nltk.download("omw-1.4")  # WordNet мовні дані

# Read the CSV file
df = pd.read_csv("assets/products_comments_list.csv")

coments = df["Comment Text"].tolist()

print("=== Токенізація ===")
sentences = sent_tokenize(" ".join(coments))
print("Речення:", sentences)

print("\n=== Видалення стоп-слів ===")
stop_words = set(stopwords.words("english"))  
words = word_tokenize(" ".join(coments))
filtered_words = [
    word for word in words if word.lower() not in stop_words and word.isalpha()
]
print("Без стоп-слів:", filtered_words)


print("\n=== Стеммінг ===")
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("Після стеммінгу:", stemmed_words)


print("\n=== Лемматизація ===")
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in filtered_words]
print("Після лемматизації:", lemmatized_words)

#2
from textblob import TextBlob

print("\n=== Аналіз тональності ===")
for comment in coments:
    blob = TextBlob(comment)
    sentiment = blob.sentiment
    print(f"Коментар: {comment}\nТональність: {sentiment}\n")


positive_comments_count = sum(1 for comment in coments if TextBlob(comment).sentiment.polarity > 0.5)
negative_comments_count = sum(1 for comment in coments if TextBlob(comment).sentiment.polarity < 0.5)
print(f"Кількість позитивних коментарів: {positive_comments_count}")
print(f"Кількість негативних коментарів: {negative_comments_count}")
