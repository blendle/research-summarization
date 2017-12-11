import torch
from torch.autograd import Variable


def rnn_ranker(w2v_sentence_means,
               original_indices,
               sentences,
               output_size,
               rnn_model,
               USE_CUDA=False):

    encoder = rnn_model.encoder
    decoder = rnn_model.decoder
    attn = rnn_model.attn

    # Pick the right sentences from sentence list (to match representation matrix)
    sentences = [sentences[i] for i in original_indices]

    # Turn sentences into torch input
    embeddings = torch.from_numpy(w2v_sentence_means)
    embeddings = Variable(embeddings, volatile=True).view(-1, 300)

    # Initialize zero vector
    BOD_vector = Variable(torch.zeros((1, 300)))
    # Initialize the label prediction as uniform ditribution over output dimensions
    label_pred = Variable(torch.FloatTensor([1] * output_size) / output_size).view(1, -1)

    if USE_CUDA:
        embeddings = embeddings.cuda()
        label_pred = label_pred.cuda()
        BOD_vector = BOD_vector.cuda()

    encoder_outputs, encoder_hidden = encoder(embeddings)
    decoder_hidden = encoder_hidden

    # For decoding, add the beginning-of-document vector (which is for now just the 0-vector) to the sequence
    embeddings = torch.cat((BOD_vector, embeddings), 0)

    scores = []

    # Now we iterate over the decoder steps
    # For evaluation, we do not use teacher forcing

    for i, decoding_step in enumerate(encoder_outputs):
        output, decoder_hidden = decoder(label_pred, embeddings[i], decoder_hidden)
        label_pred, output = attn(output, encoder_outputs[i])
        score = label_pred.data[0][1]
        scores.append(score)

    sorted_sentences = sorted(((scores[i], s, original_indices[i])
                               for i, s in enumerate(sentences)),
                              reverse=True)

    return [(i, s) for (score, s, i) in sorted_sentences]
