class Lexicon():
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

    def get_positive_negative_words(self, tweet: list):
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


class ElongatedWords():
    def get_elongated_words(self, tweet: list):

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


class CharsCount():
    def get_relevant_chars(self, tweet: str):
        num_hashtags = tweet.count('#')
        num_exclamations = tweet.count('!')
        num_interrogations = tweet.count('?')
        num_at = tweet.count('@')

        return [num_hashtags, num_exclamations, num_interrogations, num_at]


class UpperCount():
    def get_upper_words(self, tweet: list):
        upper_count = 0
        for word in tweet:
            if word.isupper():
                upper_count += 1
        return [upper_count]


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lexicon = Lexicon()
        self.elongated_words = ElongatedWords()
        self.chars_count = CharsCount()
        self.upper_count = UpperCount()

    def transform(self, X, y=None):
        chars = []

        for tweet in X:
            split_tweet = tweet.split()

            lexicon = self.lexicon.get_positive_negative_words(split_tweet)

            elongated_words = self.elongated_words.get_elongated_words(
                    split_tweet
                    )

            chars_count = self.chars_count.get_relevant_chars(tweet)

            upper_count = self.upper_count.get_upper_words(split_tweet)

            vect = lexicon + elongated_words + chars_count + upper_count

            chars.append(vect)

        return np.array(chars)

    def fit(self, X, y=None):
        return self
