import tweepy

def setup_connection(consumer_key, consumer_secret, access_token, access_token_secret):
    # Twitter Credentials
    print("setting up connection...")
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
    return api

def grab_tweets(api, target_user, number_of_tweets):
    print("grabbing your tweets...")
    return api.user_timeline(target_user, count=number_of_tweets)