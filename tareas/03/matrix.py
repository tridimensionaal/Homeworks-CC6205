#!/home/tridimensional/u/dcc/noveno/nlp/nlp/bin/python3
import numpy as np
from collections import Counter
import string
import re
import pandas as pd
from gensim.models import Word2Vec 
from gensim.models.phrases import Phrases, Phraser


class WordContextMatrix:
    def __init__(self, vocab_size, window_size, dataset, tokenizer):
        """
        Utilice el constructor para definir los parametros.
        """
        self.vocab_size = vocab_size
        self.n = 0
        self.window_size = window_size
        self.vocab = {}
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.mat = np.zeros((self.vocab_size, self.vocab_size))

        self.create_vocab()
        self.build_matrix()

    def create_vocab(self):
        cleaned_content = list(map(self.tokenizer, self.dataset))
        phrases = Phrases(cleaned_content, min_count=100, progress_per=5000) 
        bigram = Phraser(phrases)
        self.dataset = bigram[cleaned_content]

        all_words = []
        for words in self.dataset:
            for word in words:
                all_words.append(word)

        count = Counter(all_words)
        count = count.most_common(self.vocab_size)

        for (word, _) in count:
            self.add_word_to_vocab(word)

    def add_word_to_vocab(self, word):
        """
        Utilice este método para agregar token
        a sus vocabulario
        """
        # Le puede ser útil considerar un token unk al vocab
        # para palabras fuera del vocab

        if self.n >= self.vocab_size:
            return
        else:
            self.vocab[word] = self.n
            self.n += 1

    def build_matrix(self):
        """
        Utilice este método para crear la palabra contexto
        """
        for words in self.dataset:
            for (i, word) in enumerate(words):
                if word in self.vocab:
                    before = i - self.window_size
                    if before >= 0:
                        for j in range(before, i):
                            if words[j] in self.vocab:
                                word_1_indx = self.vocab[word]
                                word_2_indx = self.vocab[words[j]]
                                self.mat[word_1_indx][word_2_indx] += 1
                                self.mat[word_2_indx][word_1_indx] += 1

    def matrix2dict(self):
        """
        Utilice este método para convertir la matriz a un diccionario de
        embeddings, donde las llaves deben ser los token del vocabulario y los
        embeddings los valores obtenidos de la matriz.
        """
        dic_matrix = {}
        for key in self.vocab.keys():
            dic_matrix[key] = self.mat[self.vocab[key]]
        return dic_matrix

        # se recomienda transformar la matrix a un diccionario de embedding.
        # por ejemplo {palabra1:vec1, palabra2:vec2, ...}

    def print_matrix(self):
        hola = "        "
        print(hola, end=" ")
        for key in self.vocab.keys():
            print(key, end=" ")
        print("")

        for key in self.vocab.keys():
            print(key + " "*(len(hola)-len(key)), end=" ")
            i = self.vocab[key]

            for key in self.vocab.keys():
                j = self.vocab[key]
                print(self.mat[i][j], end=" ")
            print("")


stopwords = pd.read_csv(
        'https://raw.githubusercontent.com/Alir3z4/stop-words/master/english.txt'
        ).values
stopwords = Counter(stopwords.flatten().tolist())


def simple_tokenizer(doc, lower=False):
    global stopwords
    punctuation = string.punctuation + "«»“”‘’…—"
    if lower:
        tokenized_doc = doc.translate(str.maketrans(
            '', '', punctuation)).lower().split()

    tokenized_doc = doc.translate(str.maketrans('', '', punctuation)).split()
    tokenized_doc = [
            token for token in tokenized_doc if token.lower() not in stopwords
            ]

    return tokenized_doc


def simple_tokenizer_1(doc):
    return re.findall(r"[\w']+|[.,!?;]", doc)


def main():
    corpus = [
            "I like deep learning.",
            "I like NLP.",
            "I enjoy flying."
            ]
    # word_context_mat = WordContextMatrix(100, 3, corpus, simple_tokenizer_1)
    # word_context_mat.print_matrix()

    data_file = "dialogue-lines-of-the-simpsons.zip"
    df = pd.read_csv(data_file)
    df = df.dropna().reset_index(drop=True)  # Quitar filas vacias

    content = df["spoken_words"]

    word_context_mat = WordContextMatrix(6000, 1, content, simple_tokenizer)
    print(word_context_mat.print_matrix())
    # print(word_context_mat.matrix2dict()["Marge"])


if __name__ == "__main__":
    main()
