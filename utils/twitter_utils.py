import tweepy
from datetime import datetime, timedelta
import os
import json
import random  # For demo purposes

class TwitterClient:
    def __init__(self):
        # In a real implementation, you would use actual API keys
        # For this example, we'll create mock data
        self.api_key = os.environ.get('TWITTER_API_KEY', '')
        self.api_secret = os.environ.get('TWITTER_API_SECRET', '')
        self.access_token = os.environ.get('TWITTER_ACCESS_TOKEN', '')
        self.access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', '')
        
        # Check if API keys are available
        self.has_api_keys = all([self.api_key, self.api_secret, 
                                 self.access_token, self.access_token_secret])
        
        if self.has_api_keys:
            # Initialize Tweepy API
            auth = tweepy.OAuth1UserHandler(
                self.api_key, self.api_secret, 
                self.access_token, self.access_token_secret
            )
            self.api = tweepy.API(auth)
        else:
            self.api = None
    
    def search_tweets(self, keyword, count=10, lang='en'):
        """Search for tweets containing keyword"""
        if self.has_api_keys and self.api:
            # Real implementation with Tweepy
            try:
                tweets = self.api.search_tweets(q=keyword, count=count, lang=lang, tweet_mode='extended')
                return tweets
            except Exception as e:
                print(f"Error searching tweets: {e}")
                return self._generate_mock_tweets(keyword, count)
        else:
            # Generate mock tweets for demonstration
            return self._generate_mock_tweets(keyword, count)
    
    def _generate_mock_tweets(self, keyword, count=10):
        """Generate mock tweets for demonstration"""
        # Sample text templates
        positive_templates = [
            "I absolutely love {keyword}! It makes my day every time.",
            "Just had an amazing experience with {keyword}. So happy!",
            "Can't believe how good {keyword} is. Totally recommend it!",
            "{keyword} is the best thing that happened today. Feeling blessed.",
            "Really enjoying my time with {keyword}. Brings me joy!"
        ]
        
        negative_templates = [
            "Not happy with {keyword} today. Disappointed.",
            "Why does {keyword} have to be so frustrating? Ugh.",
            "Had a bad experience with {keyword}. Would not recommend.",
            "{keyword} is overrated. Don't waste your time.",
            "So tired of dealing with {keyword} problems. Annoyed."
        ]
        
        neutral_templates = [
            "Just checking out {keyword}. Seems okay.",
            "Anyone else tried {keyword} recently? Looking for opinions.",
            "Heard about {keyword} from a friend. Might try it.",
            "{keyword} has some pros and cons. Still deciding.",
            "Thinking about {keyword} today. Not sure what to expect."
        ]
        
        # Common usernames
        usernames = [
            "user123", "techfan22", "social_butterfly", 
            "digital_nomad", "coffee_lover", "book_worm", 
            "travel_enthusiast", "food_critic", "music_fan",
            "sports_guru", "news_junkie", "art_admirer"
        ]
        
        # Create mock tweet class
        class MockTweet:
            def __init__(self, text, username, created_at):
                self.text = text
                self.user = type('obj', (object,), {'screen_name': username})
                self.created_at = created_at
        
        # Generate tweets
        tweets = []
        now = datetime.now()
        
        for i in range(count):
            # Select template category
            category = random.choice(['positive', 'negative', 'neutral'])
            
            if category == 'positive':
                template = random.choice(positive_templates)
            elif category == 'negative':
                template = random.choice(negative_templates)
            else:
                template = random.choice(neutral_templates)
            
            # Fill in the template
            text = template.format(keyword=keyword)
            
            # Select random username
            username = random.choice(usernames)
            
            # Generate random timestamp within the last 3 days
            random_hours = random.randint(0, 72)
            timestamp = now - timedelta(hours=random_hours)
            
            # Create mock tweet
            tweet = MockTweet(text, username, timestamp)
            tweets.append(tweet)
        
        return tweets