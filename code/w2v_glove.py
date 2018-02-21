import os

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from w2v import W2V


PRETRAINED_DIR = '../resources/glove.6B.50d.txt'
GENSIM_GLOVE_DIR = '../resources/glove.gensim.6B.50d.txt'
W2V_GLOVE_MODEL_DIR = '../models/glove_model.bin'
VERBOSE = False


class W2VGlove(W2V):
    def __init__(self):
        # Check if the word2vec model exists
        if os.path.exists(W2V_GLOVE_MODEL_DIR):
            self.load(W2V_GLOVE_MODEL_DIR)
        else:
            if not os.path.exists(PRETRAINED_DIR):
                raise Exception('Pretrained glove vector file not found:', PRETRAINED_DIR)
            if not os.path.exists(GENSIM_GLOVE_DIR):
                # This adds a header line containing number of vectors and dimensionality
                glove2word2vec(PRETRAINED_DIR, GENSIM_GLOVE_DIR)
            # KeyedVectors superclasses Word2Vec
            self.glove_model = KeyedVectors.load_word2vec_format(GENSIM_GLOVE_DIR, binary=False)
            self.save(W2V_GLOVE_MODEL_DIR)

    def load(self, filename):
        # to load pretrained vectors for word
        self.glove_model = KeyedVectors.load(filename)

    def save(self, filename):
        # to save model
        self.glove_model.save(filename)

    def get_word(self, word):
        if word in self.glove_model:
            return self.glove_model[word]
        else:
            if VERBOSE:
                print 'Word not found:', word
            # We don't raise an exception so that the client need not have to handle it
            return None

    def get_sentence(self, sentence):
        words = sentence.split()
        vectors = [self.get_word(word) for word in words if not self.get_word(word) is None]
        return vectors

    def train(self, emails):
        # Pretrained GloVe model is not fine-tuned (for now)
        pass


def main():
    model = W2VGlove()

    # checking loaded_model
    print(model.glove_model['word'])

    # checking for get_word
    print(model.get_word('abacadavrab'))

    # checking for get_sentence
    sentence = model.get_sentence('abacadavrab is a word that is not present in the glove model')
    print(len(sentence))

    # result = model.glove_model.most_similar(positive=['dog', 'cow'], negative=['puppy'], topn=1)
    # print(result)


if __name__ == "__main__":
    main()