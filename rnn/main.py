from gensim.models.word2vec import Word2Vec, KeyedVectors
from rnn.train import train_model
import torch

print('Loading word2vec model...', flush=True)
model_path = 'models/word2vec/blendle/word2vec_blendle'
w2v_model = Word2Vec.load(model_path).wv
# model_path = 'models/word2vec/google/GoogleNews-vectors-negative300.bin'
# w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print('Finished loading word2vec model.', flush=True)

# For loading pretraining data, replace #PRETRAIN_PATH# with model path:
# pretrain_path = #PRETRAIN_PATH#
# pretraining_parameters = torch.load(
#     pretrain_path,
#     map_location=(lambda storage, loc: storage))
pretraining_parameters = None

train_model(w2v_model=w2v_model,
            epochs=1,
            test_size=None,
            output_size=2,
            datadir='datasets/NeuralSum/data/',
            savedir='rnn/saved_models',
            cotraining_ratio=None,
            pretraining_parameters=pretraining_parameters,
            pretrain_epochs=None,
            USE_CUDA=False)
