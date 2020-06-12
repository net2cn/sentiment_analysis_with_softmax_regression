# wubba lubba dub dub
import numpy as np
import scipy.sparse as sp                               # Using scipy's sparse matrix to get better performance.
import array
import pickle
import re
import os
import tqdm                                             # I want a tiny little neat progress bar so I imported this.
from sklearn.linear_model import LogisticRegression     # For comparison here.

def read_data(path, separator):
    # Kinda generic raw-data parser.
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
        lines = f.read().splitlines()

    print("Processing raw-data...")

    if separator:
        for line in tqdm.tqdm(lines, ascii=True):
            data.append(line.split(separator))
    else:
        data = lines

    return data

def safe_sparse_dot(a, b, *, dense_output=False):
    if a.ndim > 2 or b.ndim > 2:
        if sp.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sp.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (sp.issparse(a) and sp.issparse(b)
            and dense_output and hasattr(ret, "toarray")):
        return ret.toarray()
    return ret

# Simple count vectorizer.
class CountVectorizer():
    def __init__(self, sentences, stopwords, dtype=np.int32):
        self.sentences = sentences
        self.stopwords = stopwords
        self.dtype = dtype

        self.vocabulary = {}
        self.features = []

        print("Calculating vocabulary...")
        self.vocabulary = self.tokenize(self.sentences)
        print("Making BoW...")
        self.features = self.vectorize(self.sentences)
        
    def tokenize(self, sentences):
        words = []
        for sentence in tqdm.tqdm(sentences, ascii=True):
            w = self.word_extraction(sentence)
            words.extend(w)
        
        # Using dict to speed-up indexing.
        words = {k: v for v, k in enumerate(sorted(list(set(words))))}
        return words

    def word_extraction(self, sentence):
        words = re.sub("[^\w]", " ",  sentence).split()
        cleaned_text = [w.lower() for w in words if w not in self.stopwords]
        return cleaned_text    
    
    def vectorize(self, sentences):
        # REALLY FANCY STUFF HERE. WATCH OUT WE ARE GOING FULL SPEED. ABRACADABRA!
        j_indices = []
        indptr = []

        values = array.array(str("i"))
        indptr.append(0)

        # Using scipy's sparse matrix to speed up manipulation and save stroage space.
        for sentence in tqdm.tqdm(sentences, ascii=True):
            words = self.word_extraction(sentence)
            feature_counter = {}

            for word in words:
                try:
                    # O(1) dict instead of O(n) for-loop
                    feature_idx = self.vocabulary[word]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                # Ignore words that are not in the vocabulary.
                except KeyError:
                    continue
            
            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        j_indices = np.asarray(j_indices, dtype=self.dtype)
        indptr = np.asarray(indptr, dtype=self.dtype)
        values = np.frombuffer(values, dtype=np.intc)

        # Using dense numpy matrix only causes unaccpetable large pkl file.
        X = sp.csr_matrix((values, j_indices, indptr),
                        shape=(len(indptr) - 1, len(self.vocabulary)),
                        dtype=self.dtype)
        X.sort_indices()
        return X

# A softmax regression that works with sparse matrix
class SoftmaxRegression(object):
    def __init__(self, eta=0.01, epochs=50,
                 l2=0.0,
                 minibatches=1,
                 n_classes=None,
                 random_seed=None):

        self.eta = eta
        self.epochs = epochs
        self.l2 = l2
        self.minibatches = minibatches
        self.n_classes = n_classes
        self.random_seed = random_seed

    def _fit(self, X, y, init_params=True):
        if init_params:
            if self.n_classes is None:
                self.n_classes = np.max(y) + 1
            self._n_features = X.shape[1]

            self.b_, self.w_ = self._init_params(
                weights_shape=(self._n_features, self.n_classes),
                bias_shape=(self.n_classes,),
                random_seed=self.random_seed)
            self.loss_ = []

        y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float)

        for i in tqdm.trange(self.epochs, ascii=True):
            for idx in self._yield_minibatches_idx(
                    n_batches=self.minibatches,
                    data_ary=y,
                    shuffle=True):
                # givens:
                # w_ -> n_feat x n_classes
                # b_  -> n_classes

                # net_input, softmax and diff -> n_samples x n_classes:
                net = self._net_input(X[idx], self.w_, self.b_)
                softm = self._softmax(net)
                diff = softm - y_enc[idx]
                mse = np.mean(diff, axis=0)

                # gradient -> n_features x n_classes
                grad = safe_sparse_dot(X[idx].T, diff)
                
                # update in opp. direction of the loss gradient
                self.w_ = self.w_ - (self.eta * grad + self.eta * self.l2 * self.w_)
                self.b_ = self.b_ - (self.eta * np.sum(diff, axis=0))

            # compute loss of the whole epoch
            net = self._net_input(X, self.w_, self.b_)
            softm = self._softmax(net)
            cross_ent = self._cross_entropy(output=softm, y_target=y_enc)
            loss = self._loss(cross_ent)
            self.loss_.append(loss)
        return self

    def fit(self, X, y, init_params=True):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self._fit(X=X, y=y, init_params=init_params)
        self._is_fitted = True
        return self
    
    def _predict(self, X):
        probas = self.predict_proba(X)
        return self._to_classlabels(probas)
 
    def predict(self, X):
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet.')
        return self._predict(X)

    def predict_proba(self, X):
        net = self._net_input(X, self.w_, self.b_)
        softm = self._softmax(net)
        return softm

    def _net_input(self, X, W, b):
        return (X.dot(W) + b)

    # A faaaaaaaaaaaaaaaster softmax
    def _softmax(self, z, copy=True):
        if copy:
            z = np.copy(z)
        max_prob = np.max(z, axis=1).reshape((-1, 1))
        z -= max_prob
        np.exp(z, z)
        sum_prob = np.sum(z, axis=1).reshape((-1, 1))
        z /= sum_prob
        return z

    def _cross_entropy(self, output, y_target):
        return -np.sum(np.log(output) * (y_target), axis=1)

    def _loss(self, cross_entropy):
        L2_term = self.l2 * np.sum(safe_sparse_dot(self.w_.T, self.w_))
        cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def _to_classlabels(self, z):
        return z.argmax(axis=1)
    
    def _init_params(self, weights_shape, bias_shape=(1,), dtype='float64',
                     scale=0.01, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        w = np.random.normal(loc=0.0, scale=scale, size=weights_shape)
        b = np.zeros(shape=bias_shape)
        return b.astype(dtype), w.astype(dtype)
    
    def _one_hot(self, y, n_labels, dtype):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)    
    
    def _yield_minibatches_idx(self, n_batches, data_ary, shuffle=True):
            indices = np.arange(data_ary.shape[0])

            if shuffle:
                indices = np.random.permutation(indices)
            if n_batches > 1:
                remainder = data_ary.shape[0] % n_batches

                if remainder:
                    minis = np.array_split(indices[:-remainder], n_batches)
                    minis[-1] = np.concatenate((minis[-1],
                                                indices[-remainder:]),
                                               axis=0)
                else:
                    minis = np.array_split(indices, n_batches)

            else:
                minis = (indices,)

            for idx_batch in minis:
                yield idx_batch
    
    def _shuffle_arrays(self, arrays):
        """Shuffle arrays in unison."""
        r = np.random.permutation(len(arrays[0]))
        return [ary[r] for ary in arrays]

def main():
    # Define paths.
    train_dataset_path = "./datasets/train.tsv"
    test_dataset_path = "./datasets/test.tsv"
    test_ground_truth_path = "./datasets/sampleSubmission.csv"
    stopwords_path = "./stopwords.txt"

    print("Reading data...")

    # Read datasets.
    train_dataset = read_data(train_dataset_path, "	")[1:]          # Remove title row.
    test_dataset = read_data(test_dataset_path, "	")[1:]          # Same.
    stopwords = read_data(stopwords_path, "")

    # Check if we have made our delicious pickles
    if not os.path.exists("./data") or not os.path.exists("./data/vectorizer.pkl"):
        os.mkdir("./data")
        print("Preparing data... This may take a while...")
        vectorizer = CountVectorizer([train_dataset[i][2] for i in range(len(train_dataset))], stopwords)
        pickle.dump(vectorizer, open("./data/vectorizer.pkl", "wb"))

    vectorizer = pickle.load(open("./data/vectorizer.pkl", "rb"))

    # Prepare dataset.
    # I'm getting lazy here. No fancy stuff.
    x_train = vectorizer.features
    y_train = np.array([int(train_dataset[i][3]) for i in range(len(train_dataset))])

    x_test = x_train[0:10000]       # Reserve some data for validating.
    y_test = y_train[0:10000]       # Note: This is for making submission. Normally you will want to
                                    #       split train-test set. See comments below.

    # x_test = x_train[0:10000]    # Reserve some data for validating.
    # y_test = y_train[0:10000]

    # x_train = x_train[10000:]
    # y_train = y_train[10000:]

    # This is for predicting test set for submission.
    x_sub = vectorizer.vectorize([test_dataset[i][2] for i in range(len(test_dataset))])

    print("Train set shape: " + str(np.shape(x_train)))
    print("Test set shape: " + str(np.shape(x_test)))

    print("Calculating logistic regression...")

    # With sklearn I'm getting a score of 0.6271.
    print("Using sklearn: ")
    model = LogisticRegression(solver='liblinear',random_state=0)
    model.fit(x_train, y_train)
    print("Sklearn accuracy: " + str(model.score(x_test, y_test)))

    # With implementation I'm getting a score of 0.6434.
    # You can of course load the pkl you dumped last time. I'm getting lazy here.
    print("Using implementation: ")
    classifier = SoftmaxRegression(eta=0.01, epochs=100, minibatches=1000, random_seed=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("Implemention accuracy: " + str(np.asarray(y_pred == y_test).mean()))

    # Save model for furthur usage.
    if not os.path.exists("./model"):
        os.mkdir("./model")

    pickle.dump(classifier, open("./model/model.pkl", "wb"))

    y_sub = classifier.predict(x_sub)

    # Dump submission csv.
    # Really stupid approach.
    csv = []
    csv.append("PhraseId,Sentiment")
    i = 0
    for line in test_dataset:
        pharse_id = line[0]
        prediction = y_pred[i]

        csv.append("{0},{1}".format(pharse_id, prediction))

        i += 1
    
    csv.append("")
    with open("submission.csv", "w") as outfile:
        outfile.write("\n".join(csv))

if __name__ == "__main__":
    main()