import codecs
import numpy as np
from abc import ABC, abstractmethod
import nltk


class CustomTokenizer:
    def __init__(self):

class Feature(ABC):
    @abstractmethod
    def get_feature(self, tweet: list):
        pass


class Lexicon(Feature):
    def __init__(self):
        self.lists_words = {}
        self.create_list_words("positive")
        self.create_list_words("negative")

    def create_list_words(self, word_type: str):
        with codecs.open(
                f"./opinion-lexicon-English/{word_type}-words.txt",
                "r",
                "iso-8859-1"
                ) as f:
            list_words = map(lambda x: x.replace("\n", ""), f.readlines())
        self.lists_words[word_type] = list(list_words)

    # override
    def get_feature(self, tweet: list):
        negative_words = self.lists_words["negative"]
        positive_words = self.lists_words["positive"]

        negatives = 0
        positives = 0
        for word in tweet:
            if word in negative_words:
                negatives += 1
            if word in positive_words:
                positives += 1

        return [positives, negatives]


class ElongatedWords(Feature):
    # override
    def get_feature(self, tweet: list):

        elongated_words = 0

        for word in tweet:
            current_letter = word[0]
            count = -1
            for letter in word:
                if letter == current_letter:
                    count += 1
                else:
                    if count >= 4:
                        elongated_words += 1
                        current_letter = letter
                        count = 0

        return [elongated_words]


class CharsCount(Feature):
    # override
    def get_feature(self, tweet: list):
        num_hashtags = 0
        num_exclamations = 0
        num_interrogations = 0
        num_at = 0

        for word in tweet:
            num_hashtags += word.count('#')
            num_exclamations += word.count('!')
            num_interrogations += word.count('?')
            num_at += word.count('@')

        return [num_hashtags, num_exclamations, num_interrogations, num_at]


class UpperCount(Feature):
    # override
    def get_feature(self, tweet: list):
        upper_count = 0
        for word in tweet:
            if word.isupper():
                upper_count += 1
        return [upper_count]


class CustomTransformer(ABC, BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = tokenizer

    def transform(self, X, y=None):
        chars = []

        tokenizer = nltk.TweetTokenizer()

        for tweet in X:
            features = []
            split_tweet = tokenizer.tokenize(tweet)

            for class_ in self.classes:
                feature = class_().get_feature(split_tweet)
                features += feature

            chars.append(features)

        return np.array(chars)

    def fit(self, X, y=None):
        return self


class AngryTransformer(CustomTransformer):
    def __init__(self):
        super(CustomTransformer, self).__init__

        self.classes = [
                Lexicon,
                ElongatedWords,
                CharsCount,
                UpperCount
                ]
