import json
from typing import Tuple
import pandas as pd
from textblob import TextBlob
from clean_tweets_dataframe import Clean_Tweets


def read_json(json_file: str) -> list:
    """
    json file reader to open and read json files into a list
    Args:
    -----
    json_file: str - path of a json file

    Returns
    -------
    length of the json file and a list of json
    """

    tweets_data = []
    for tweets in open(json_file, 'r'):
        tweets_data.append(json.loads(tweets))

    return len(tweets_data), tweets_data


class TweetDfExtractor:
    """
    this function will parse tweets json into a pandas dataframe

    Return
    ------
    dataframe
    """

    def __init__(self, tweets_list):

        self.tweets_list = tweets_list

    # an example function

    def find_statuses_count(self) -> list:
        statuses_count = [tweets['user']['statuses_count']
                          for tweets in self.tweets_list]

        return statuses_count

    def find_full_text(self) -> list:
        full_text = [tweets["full_text"] for tweets in self.tweets_list]
        return full_text

    def cleaned_tweet(self, pd_text: pd.DataFrame) -> pd.DataFrame:
        # text = Clean_Tweets()
        pd_text['cleaned_text'] = pd_text['original_text'].str.replace(
            r'http\S+', '', regex=True)
        pd_text['cleaned_text'] = pd_text['cleaned_text'].str.replace(
            r'@\w+', '', regex=True)
        pd_text['cleaned_text'] = pd_text['cleaned_text'].str.replace(
            r'#', '', regex=True)
        pd_text['cleaned_text'] = pd_text['cleaned_text'].str.replace(
            r'RT', '', regex=True)
        pd_text['cleaned_text'] = pd_text['cleaned_text'].str.replace(
            r':', '', regex=True)
        pd_text['cleaned_text'] = pd_text['cleaned_text'].str.replace(
            r'\n', '')

        return pd_text

    def find_sentiments(self, text: pd.Series) -> Tuple[list, list]:
        def get_polarity(text):
            analysis = TextBlob(text)
            return analysis.sentiment.polarity

        def get_subjectivity(text):
            analysis = TextBlob(text)
            return analysis.sentiment.subjectivity
        self.polarity = [get_polarity(t) for t in text]
        self.subjectivity = [get_subjectivity(t) for t in text]

        return self.polarity, self.subjectivity

    def find_created_time(self) -> list:
        created_at = [tweets['created_at'] for tweets in self.tweets_list]
        return created_at

    def find_source(self) -> list:
        source = [tweets['source'] for tweets in self.tweets_list]
        return source

    def find_screen_name(self) -> list:
        screen_name = [tweets['user']['screen_name']
                       for tweets in self.tweets_list]
        return screen_name

    def find_followers_count(self) -> list:
        followers_count = [tweets['user']['followers_count']
                           for tweets in self.tweets_list]
        return followers_count

    def find_friends_count(self) -> list:
        friends_count = [tweets['user']['friends_count']
                         for tweets in self.tweets_list]
        return friends_count

    def is_sensitive(self) -> list:
        try:
            is_sensitive = [x['possibly_sensitive'] for x in self.tweets_list]
        except KeyError:
            is_sensitive = ['false']*len(self.tweets_list)

        return is_sensitive

    def find_favourite_count(self) -> list:
        favourite_count = [tweets['favorite_count']
                           for tweets in self.tweets_list]
        return favourite_count

    def find_retweet_count(self) -> list:
        retweet_count = [tweets['retweet_count']
                         for tweets in self.tweets_list]
        return retweet_count

    def find_hashtags(self) -> list:
        hashtags = [tweets['entities']['hashtags']
                    for tweets in self.tweets_list]
        return hashtags

    def find_mentions(self) -> list:
        mentions = [tweets['entities']['user_mentions']
                    for tweets in self.tweets_list]
        return mentions

    def find_location(self) -> list:
        try:
            location = self.tweets_list['user']['location']
        except TypeError:
            location = [""]*len(self.tweets_list)

        return location

    def find_lang(self) -> list:
        lang = [tweets['lang'] for tweets in self.tweets_list]
        return lang

    def get_tweet_df(self, save=False) -> pd.DataFrame:
        """required column to be generated you should be creative and add more features"""

        columns = ['created_at', 'source',
                   'original_text', 'polarity', 'subjectivity', 'lang', 'fav_count', 'retweet_count', 'screen_name',
                   'follower_count', 'friends_count', 'sensitivity', 'hashtags', 'mentions', 'location']

        created_at = self.find_created_time()
        source = self.find_source()
        text = self.find_full_text()
        polarity, subjectivity = self.find_sentiments(text)
        lang = self.find_lang()
        fav_count = self.find_favourite_count()
        retweet_count = self.find_retweet_count()
        screen_name = self.find_screen_name()
        follower_count = self.find_followers_count()
        friends_count = self.find_friends_count()
        sensitivity = self.is_sensitive()
        hashtags = self.find_hashtags()
        mentions = self.find_mentions()
        location = self.find_location()
        # print(len(created_at), len(source), len(text), len(polarity), len(subjectivity), len(lang), len(fav_count), len(retweet_count), len(
        #     screen_name), len(follower_count), len(friends_count), len(sensitivity), len(hashtags), len(mentions), len(location))
        data = zip(created_at, source, text, polarity, subjectivity,
                   lang, fav_count, retweet_count, screen_name, follower_count, friends_count, sensitivity, hashtags, mentions, location)
        df = pd.DataFrame(data=data, columns=columns)

        if save:
            df.to_csv('processed_tweet_data.csv', index=False)
            print('File Successfully Saved.!!!')

        return df


if __name__ == "__main__":
    # required column to be generated you should be creative and add more features
    columns = ['created_at', 'source', 'original_text', 'clean_text', 'sentiment', 'polarity', 'subjectivity', 'lang', 'favorite_count', 'retweet_count',
               'original_author', 'screen_count', 'followers_count', 'friends_count', 'possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']
    _, tweets_list = read_json("test_data_100.json")

    tweet = TweetDfExtractor(tweets_list)
    tweet_df = tweet.get_tweet_df(save=True)

    # As the input for cleaned_tweet is a pandaframe, I had to excute cleaned_tweet, polarity and
    # subjectivity of cleaned files separately outside

    tweet_df = tweet.cleaned_tweet(tweet_df)

    print(tweet_df['subjectivity'][0:5])
    polarity_cleaned, subjectivity_cleaned = tweet.find_sentiments(
        tweet_df['cleaned_text'])
    tweet_df['polarity_cleaned'] = polarity_cleaned
    tweet_df['subjectivity_cleaned'] = subjectivity_cleaned

    # Now I can save everything as a processesd csv file
    tweet_df.to_csv('processed_and_cleaned_tweet_data.csv', index=False)

    # use all defined functions to generate a dataframe with the specified columns above
