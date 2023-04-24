import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob

# Define search query and date range
query = "crypto OR cryptocurrency OR bitcoin OR ethereum"
since_date = "2022-01-01" #start date
until_date = "2023-03-31" #end date

# Use snscrape to scrape tweets
tweets = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f"{query} since:{since_date} until:{until_date}").get_items()):
    if i > 1000:  # limit to 1000 tweets
        break
    tweets.append([tweet.date, tweet.content])

# Convert scraped tweets into a pandas dataframe
df = pd.DataFrame(tweets, columns=["date", "text"])

# Define a function to classify the sentiment of a given text as positive, negative, or neutral
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# Apply sentiment classification to each tweet
df["sentiment"] = df["text"].apply(get_sentiment)

# Save the dataset to a CSV file
df.to_csv("cryptocurrency_tweets.csv", index=False)
