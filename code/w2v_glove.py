from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec


class w2v:

	#input and output vector file name
	input_glove_trained_vectors = 'resources/glove50d.txt'
	output_glove2word2vec = 'resources/glove2word2vec50d'

	def __init__(self):
		#converting to word2vec format
		glove2word2vec(self.input_glove_trained_vectors, self.output_glove2word2vec)

	def load(self, filename):
		#to load pretrained vectors for word
		self.glove_model = KeyedVectors.load_word2vec_format(filename, binary=False)
		return self.glove_model

# result = model.most_similar(positive=['dog', 'cow'], negative=['puppy'], topn=1)
# print(result)

	def save(self, filename):
		#to save model
		self.glove_model.save(filename)

	def get_word(self, word):
		return self.glove_model[word]

#to load already saved model
# loaded_model = Word2Vec.load('glove_model.bin')
# print(loaded_model['sentence'])

def main():

	GloVe_model = w2v()

	#loading model
	loaded_model = GloVe_model.load('resources/glove2word2vec50d')

	#checking loaded_model
	print(loaded_model['word'])

	#saving model
	GloVe_model.save('models/glove_model.bin')

	#checking for get_word 
	try:
		print(GloVe_model.get_word('abacadavrab'))
	except KeyError:
		print('Word Not Found!!!')

if __name__ == "__main__":
	main()
	
