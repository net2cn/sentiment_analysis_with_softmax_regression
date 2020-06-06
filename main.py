import numpy as np
import scipy.sparse as sp
import array
import pickle
import re
import os
import tqdm

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
    def __init__(self, sentences, stopwords, dtype=np.int64):
        self.sentences = sentences
        self.stopwords = stopwords
        self.dtype = dtype

        self.vocabulary = {}
        self.bow = []
        
    def tokenize(self, sentences):
        words = []
        for sentence in tqdm.tqdm(sentences, total=len(sentences), ascii=True):
            w = self.word_extraction(sentence)
            words.extend(w)
        
        words = {k: v for v, k in enumerate(sorted(list(set(words))))}
        return words

    def word_extraction(self, sentence):
        words = re.sub("[^\w]", " ",  sentence).split()
        cleaned_text = [w.lower() for w in words if w not in self.stopwords]
        return cleaned_text    
        
    def vectorize(self, sentences, vocab):
        j_indices = []
        indptr = []

        values = array.array(str("i"))
        indptr.append(0)

        # Partially from scipy. For significant speed-up.
        for sentence in tqdm.tqdm(sentences, total=len(sentences), ascii=True):
            words = self.word_extraction(sentence)
            feature_counter = {}
            for word in words:
                try:
                    feature_idx = vocab[word]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    continue
            
            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            if _IS_32BIT:
                raise ValueError(('sparse CSR array has {} non-zero '
                                'elements and requires 64 bit indexing, '
                                'which is unsupported with 32 bit Python.')
                                .format(indptr[-1]))
                indices_dtype = np.int64
        else:
            indices_dtype = np.int32

        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        # Using dense matrix only causes unaccpetable large pkl file.
        X = sp.csr_matrix((values, j_indices, indptr),
                        shape=(len(indptr) - 1, len(vocab)),
                        dtype=self.dtype)
        X.sort_indices()
        return X

    def run(self):
        print("Calculating vocabulary...")
        self.vocabulary = self.tokenize(self.sentences)
        print("Making BoW...")
        self.bow = self.vectorize(self.sentences, self.vocabulary)

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

    if not os.path.exists("./data/vectorizer.pkl"):
        print("Preparing data... This may take a while...")
        train_vectorizer = CountVectorizer([train_dataset[i][2] for i in range(len(train_dataset))], stopwords)
        train_vectorizer.run()
        os.mkdir("./data")
        pickle.dump(train_vectorizer, open("./data/vectorizer.pkl", "wb"))

    train_vectorizer = pickle.load(open("./data/vectorizer.pkl", "rb"))

    x_train = train_vectorizer.bow
    y_train = [train_dataset[i][3] for i in range(len(train_dataset))]





if __name__ == "__main__":
    main()