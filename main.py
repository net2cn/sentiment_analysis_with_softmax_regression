import numpy as np
import pickle
import re
import os
import multiprocessing

def read_data(path):
    # PhraseId SentenceId Phrase Sentiment
    # 0 1 2 3
    
    # The sentiment labels are:
    # 0 - negative
    # 1 - somewhat negative
    # 2 - neutral
    # 3 - somewhat positive
    # 4 - positive
    data = []

    with open(path, "r") as f:
        lines = f.read().splitlines()[1:]   # Remove header.

    for line in lines:
        data.append(line.split("	"))

    return data

class CountVectorizer():
    def __init__(self, sentences, stopwords):
        self.sentences = sentences
        self.stopwords = stopwords

        self.vocabulary = []
        self.bow = []

        self.vocabulary = self.tokenize(self.sentences)
        self.bow = self.generate_bow(self.sentences, self.vocabulary)
        
    def tokenize(self, sentences):
        words = []
        for sentence in sentences:
            w = self.word_extraction(sentence)
            words.extend(w)
        
        words = sorted(list(set(words)))
        return words

    def word_extraction(self, sentence):
        words = re.sub("[^\w]", " ",  sentence).split()
        cleaned_text = [w.lower() for w in words if w not in self.stopwords]
        return cleaned_text    
        
    def generate_bow(self, sentences, vocab):
        bow = []

        for sentence in sentences:
            words = self.word_extraction(sentence)
            bag_vector = np.zeros(len(vocab))
            for w in words:
                for i,word in enumerate(vocab):
                    if word == w: 
                        bag_vector[i] += 1
            
            bow.append(np.array(bag_vector))

        return bow

def LogisticRegressor():
    def __init__(x, y):
        self.x = x
        self.y = y

        self.weights = np.zeros()
        self.bias = np.zeros()
        self.learning_rate = 0.01
        self.epoches = 5000

    def relu(x):
        return np.maximum(x, 0)


def main():
    train_dataset_path = "./datasets/train.tsv"
    test_dataset_path = "./datasets/test.tsv"
    stopwords_path = "./stopwords.txt"

    train_dataset = read_data(train_dataset_path)[1:]   # Remove title row.
    test_dataset = read_data(test_dataset_path)[1:]
    stopwords = sum(read_data(stopwords_path), [])  # Flatten stopwords

    train_vectorizer = CountVectorizer([train_dataset[i][2] for i in range(len(train_dataset))][0:50], stopwords)
    print(*train_vectorizer.bow, sep="\n")

    # if not os.path.exists("./data/train.pkl"):
    #     print("Preparing data... This may take a while...")
    #     train_vectorizer = CountVectorizer([train_dataset[i][2] for i in range(len(train_dataset))][0:50], stopwords)
    #     pickle.dump(train_vectorizer)

    # train_vectorizer = pickle.load("./data/train.pkl")

    # x_train = train_vectorizer.bow
    # y_train = [train_dataset[i][3] for i in range(len(train_dataset))]





if __name__ == "__main__":
    main()