from tqdm import tqdm
import numpy as np
import random
import itertools
import json
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from rnn.data import DataIterator
from rnn.model import EncoderRNN, DecoderRNN, Attention


def train_model(w2v_model,
                epochs,
                output_size,
                datadir,
                savedir,
                cotraining_ratio=None,
                pretrain_epochs=None,
                pretraining_parameters=None,
                test_size=None,
                USE_CUDA=False):

    if pretrain_epochs and pretraining_parameters:
        raise ValueError('Both pretrain_epochs and pretraining_parameter arguments are given; '
                         'at least one of them should be falsy.')

    print('Current time: {}'.format(time.strftime('%l:%M%p | %b %d')))
    print('Loading data...', flush=True)
    # Load data
    data = {}
    for name in ('train', 'valid', 'test'):
        data[name] = DataIterator(dirname=datadir, w2v_model=w2v_model,
                                  dataset_name=name)

    if test_size:
        print("Loading data chunks of {}".format(test_size), flush=True)
        train_data_subset = list(itertools.islice(data['train'], test_size))
        valid_data_subset = list(itertools.islice(data['valid'], test_size))
        test_data_subset = list(itertools.islice(data['test'], test_size))
        train_data = DataLoader(train_data_subset, num_workers=4, shuffle=True)
        valid_data = DataLoader(valid_data_subset, num_workers=4, shuffle=False)
        test_data = DataLoader(test_data_subset, num_workers=4, shuffle=False)
    else:
        train_data = DataLoader(list(data['train']), num_workers=4, shuffle=True)
        valid_data = DataLoader(list(data['valid']), num_workers=4, shuffle=False)
        test_data = DataLoader(list(data['test']), num_workers=4, shuffle=False)

    print('Finished loading data.')
    print('Subset lengths: {}, {}, {}'.format(len(train_data),
                                              len(valid_data),
                                              len(test_data)), flush=True)

    # Initialize model & training parameters
    teacher_forcing_ratio = 0.5
    embedding_size = 300
    rnn_hidden_size = 500
    attn_hidden_size = 500
    dropout_p = .3
    lr = 1e-3

    encoder = EncoderRNN(embedding_size, rnn_hidden_size, dropout_p=dropout_p)
    attn = Attention(rnn_hidden_size, attn_hidden_size, output_size)
    decoder = DecoderRNN(embedding_size + output_size, rnn_hidden_size, dropout_p=dropout_p)

    # Define loss criterion
    criterion = torch.nn.NLLLoss()

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        attn = attn.cuda()

    loss_means = []
    accuracy_means = []
    loss_save_interval = 1000

    if pretrain_epochs:

        # Use separately initialized optimizers for pretraining
        encoder_optim = torch.optim.Adam(encoder.parameters(), lr=lr)
        decoder_optim = torch.optim.Adam(decoder.parameters(), lr=lr)
        attn_optim = torch.optim.Adam(attn.parameters(), lr=lr)
        optims = [encoder_optim, decoder_optim, attn_optim]

        for epoch in range(pretrain_epochs):
            print('\nPretraining epoch {}'.format(epoch + 1))
            print('Current time: {}'.format(time.strftime('%l:%M%p | %b %d')))

            # Set model to train mode
            encoder.train()
            decoder.train()
            attn.train()

            # Bookkeeping
            losses = []
            loss_sum = 0
            accuracy_sum = 0

            accuracies = []
            pbar = tqdm(train_data, total=len(train_data))

            # Start training
            for iteration, (sentences, labels, textrank_labels) in enumerate(pbar):
                # Perform a train step
                loss, label_preds = train_step(sentences=sentences,
                                               labels=textrank_labels,
                                               encoder=encoder,
                                               decoder=decoder,
                                               attn=attn,
                                               optims=optims,
                                               criterion=criterion,
                                               teacher_forcing_ratio=teacher_forcing_ratio,
                                               output_size=output_size,
                                               USE_CUDA=USE_CUDA)

                accuracy = (label_preds == labels[0].numpy()).mean()
                accuracies.append(accuracy)
                accuracy_sum += accuracy
                losses.append(loss)
                loss_sum += loss

                mean_interval = 1000
                if iteration <= mean_interval:
                    pbar.set_postfix({'mean loss/accuracy': '{:.4f}, {:.4f}'.format(np.mean(losses),
                                                                                    np.mean(
                                                                                        accuracies))})
                if iteration > mean_interval:
                    pbar.set_postfix({'mean loss/accuracy (last {})'.format(mean_interval):
                        '{:.4f}, {:.4f}'.format(
                            np.mean(losses[-(mean_interval + 1):-1]),
                            np.mean(accuracies[-(mean_interval + 1):-1]))})

                if iteration % loss_save_interval == 0 and iteration != 0:
                    loss_mean = loss_sum / loss_save_interval
                    loss_means.append(loss_mean)
                    accuracy_mean = accuracy_sum / loss_save_interval
                    accuracy_means.append(accuracy_mean)
                    loss_sum = 0
                    accuracy_sum = 0

            tqdm.write('Train accuracy: {:.4f}\t\tTrain loss: {:.4f}'.format(np.mean(accuracies),
                                                                             np.mean(losses)))

            # Set model to eval mode
            encoder.eval()
            decoder.eval()
            attn.eval()

            # Reset bookkeeping
            losses = []
            accuracies = []

            # Start evaluation on validation set
            for iteration, (sentences, labels, textrank_labels) in enumerate(valid_data):
                # Perform an eval step
                loss, label_preds = eval_step(sentences=sentences,
                                              labels=textrank_labels,
                                              encoder=encoder,
                                              decoder=decoder,
                                              attn=attn,
                                              criterion=criterion,
                                              output_size=output_size,
                                              USE_CUDA=USE_CUDA)

                accuracy = (label_preds == labels[0].numpy()).mean()
                accuracies.append(accuracy)
                losses.append(loss)

            tqdm.write(
                'Validation accuracy: {:.4f}\tValidation loss: {:.4f}'.format(np.mean(accuracies),
                                                                              np.mean(losses)))

            # Save ALL THE THINGS
            model_save_path = savedir + 'rnn_model_state_pretrain_epoch' + str((epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'final': False,
                'enc_state_dict': encoder.state_dict(),
                'dec_state_dict': decoder.state_dict(),
                'att_state_dict': attn.state_dict(),
                'optim_state_dicts': [optim.state_dict() for optim in optims]
            }, model_save_path)
            # Save losses/accuracies
            losses_accuracies = {'losses': loss_means,
                                 'accuracies': accuracy_means}
            with open(savedir + 'loss_accuracy.json', 'w') as fh:
                json.dump(losses_accuracies, fh)
            print('Saved current model state & losses/accuracies.')

    # And now, for the real training:
    # If parameters from pretraining are given, load them into model
    if pretraining_parameters:
        print('Using parameters from pretraining.', flush=True)
        encoder.load_state_dict(pretraining_parameters['enc_state_dict'])
        decoder.load_state_dict(pretraining_parameters['dec_state_dict'])
        attn.load_state_dict(pretraining_parameters['att_state_dict'])

    # Always re-initialize the optimizers
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=lr)
    attn_optim = torch.optim.Adam(attn.parameters(), lr=lr)
    optims = [encoder_optim, decoder_optim, attn_optim]

    for epoch in range(epochs):
        print('\nEpoch {}'.format(epoch + 1))
        print('Current time: {}'.format(time.strftime('%l:%M%p | %b %d')))

        # Set model to train mode
        encoder.train()
        decoder.train()
        attn.train()

        # Bookkeeping
        losses = []
        loss_sum = 0
        accuracy_sum = 0

        accuracies = []
        pbar = tqdm(train_data, total=len(train_data))

        # Start training
        for iteration, (sentences, labels, textrank_labels) in enumerate(pbar):
            # If a cotraining ratio is given, use textrank labels randomly according to ratio.
            if cotraining_ratio:
                use_textrank_labels = random.random() < cotraining_ratio
                if use_textrank_labels:
                    labels = textrank_labels

            # Perform a train step
            loss, label_preds = train_step(sentences=sentences,
                                           labels=labels,
                                           encoder=encoder,
                                           decoder=decoder,
                                           attn=attn,
                                           optims=optims,
                                           criterion=criterion,
                                           teacher_forcing_ratio=teacher_forcing_ratio,
                                           output_size=output_size,
                                           USE_CUDA=USE_CUDA)

            accuracy = (label_preds == labels[0].numpy()).mean()
            accuracies.append(accuracy)
            accuracy_sum += accuracy
            losses.append(loss)
            loss_sum += loss

            mean_interval = 1000
            if iteration <= mean_interval:
                pbar.set_postfix({'mean loss/accuracy': '{:.4f}, {:.4f}'.format(np.mean(losses),
                                                                                 np.mean(
                                                                                     accuracies))})
            if iteration > mean_interval:
                pbar.set_postfix({'mean loss/accuracy (last {})'.format(mean_interval):
                                      '{:.4f}, {:.4f}'.format(
                                          np.mean(losses[-(mean_interval + 1):-1]),
                                          np.mean(accuracies[-(mean_interval + 1):-1]))})

            if iteration % loss_save_interval == 0 and iteration != 0:
                loss_mean = loss_sum / loss_save_interval
                loss_means.append(loss_mean)
                accuracy_mean = accuracy_sum / loss_save_interval
                accuracy_means.append(accuracy_mean)
                loss_sum = 0
                accuracy_sum = 0

        tqdm.write('Train accuracy: {:.4f}\t\tTrain loss: {:.4f}'.format(np.mean(accuracies),
                                                                             np.mean(losses)))

        # Set model to eval mode
        encoder.eval()
        decoder.eval()
        attn.eval()

        # Reset bookkeeping
        losses = []
        accuracies = []

        # Start evaluation on validation set
        for iteration, (sentences, labels, textrank_labels) in enumerate(valid_data):
            # Perform an eval step
            loss, label_preds = eval_step(sentences=sentences,
                                          labels=labels,
                                          encoder=encoder,
                                          decoder=decoder,
                                          attn=attn,
                                          criterion=criterion,
                                          output_size=output_size,
                                          USE_CUDA=USE_CUDA)

            accuracy = (label_preds == labels[0].numpy()).mean()
            accuracies.append(accuracy)
            losses.append(loss)

        tqdm.write('Validation accuracy: {:.4f}\tValidation loss: {:.4f}'.format(np.mean(accuracies),
                                                                                 np.mean(losses)))

        # Save ALL THE THINGS
        model_save_path = savedir + 'rnn_model_state_epoch' + str((epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'final': False,
            'enc_state_dict': encoder.state_dict(),
            'dec_state_dict': decoder.state_dict(),
            'att_state_dict': attn.state_dict(),
            'optim_state_dicts': [optim.state_dict() for optim in optims]
        }, model_save_path)
        # Save losses/accuracies
        losses_accuracies = {'losses': loss_means,
                             'accuracies': accuracy_means}
        with open(savedir + 'loss_accuracy.json', 'w') as fh:
            json.dump(losses_accuracies, fh)
        print('Saved current model state & losses/accuracies.')


    # Set model to eval mode
    encoder.eval()
    decoder.eval()
    attn.eval()

    # Reset bookkeeping
    losses = []
    accuracies = []

    # Start evaluation on validation set
    for iteration, (sentences, labels, textrank_labels) in enumerate(test_data):
        # Perform an eval step
        loss, label_preds = eval_step(sentences=sentences,
                                      labels=labels,
                                      encoder=encoder,
                                      decoder=decoder,
                                      attn=attn,
                                      criterion=criterion,
                                      output_size=output_size,
                                      USE_CUDA=USE_CUDA)

        accuracy = (label_preds == labels[0].numpy()).mean()
        accuracies.append(accuracy)
        losses.append(loss)

    print('\nTest accuracy: {:.4f}\t\tTest loss: {:.4f}'.format(np.mean(accuracies), np.mean(losses)))
    print('\nSaving final model state & losses/accuracies...')
    # Save ALL THE THINGS
    model_save_path = savedir + 'rnn_model_state_final'
    torch.save({
        'epoch': None,
        'final': True,
        'enc_state_dict': encoder.state_dict(),
        'dec_state_dict': decoder.state_dict(),
        'att_state_dict': attn.state_dict(),
        'optim_state_dicts': [optim.state_dict() for optim in optims]
    }, model_save_path)

    # Save losses/accuracies
    losses_accuracies = {'losses': loss_means,
                         'accuracies': accuracy_means}
    with open(savedir + 'loss_accuracy.json', 'w') as fh:
        json.dump(losses_accuracies, fh)

    print('Finished saving!')
    print('Finish time: {}'.format(time.strftime('%l:%M%p | %b %d')))


def eval_step(sentences,
              labels,
              encoder,
              decoder,
              attn,
              criterion,
              output_size,
              USE_CUDA):
    # Turn x and y into Variables
    sentences = Variable(sentences, volatile=True).view(-1, sentences.size(2))
    labels = Variable(torch.LongTensor(labels), volatile=True).view(-1)

    # Initialize zero vector
    BOD_vector = Variable(torch.zeros((1, 300)))
    # Initialize the label prediction as uniform ditribution over output dimensions
    label_pred = Variable(torch.FloatTensor([1] * output_size) / output_size).view(1, -1)
    # Initialize loss
    loss = Variable(torch.zeros(1))

    if USE_CUDA:
        sentences = sentences.cuda()
        labels = labels.cuda()
        label_pred = label_pred.cuda()
        loss = loss.cuda()
        BOD_vector = BOD_vector.cuda()

    encoder_outputs, encoder_hidden = encoder(sentences)
    decoder_hidden = encoder_hidden

    # For decoding, add the beginning-of-document vector (which is for now just the 0-vector) to the sequence
    sentences = torch.cat((BOD_vector, sentences), 0)

    label_preds = np.zeros((len(encoder_outputs))).astype(int)

    # Now we iterate over the decoder steps
    # For evaluation, we do not use teacher forcing
    for i, decoding_step in enumerate(encoder_outputs):
        output, decoder_hidden = decoder(label_pred, sentences[i], decoder_hidden)
        label_pred, output = attn(output, encoder_outputs[i])
        loss += criterion(output, labels[i])
        label_preds[i] = label_pred.max(-1)[1].data[0]

    # Compute mean loss
    loss /= len(encoder_outputs)

    return loss.data[0], label_preds


def train_step(sentences,
               labels,
               encoder,
               decoder,
               attn,
               optims,
               criterion,
               teacher_forcing_ratio,
               output_size,
               USE_CUDA):
    # Turn x and y into Variables
    sentences = Variable(sentences).view(-1, sentences.size(2))
    labels = Variable(torch.LongTensor(labels)).view(-1)

    # Make labels one-hot and append 0-label, for teacher forcing input
    label_list = labels.data.numpy()
    true_labels = np.zeros((len(label_list), output_size))
    true_labels[np.arange(len(label_list)), label_list] = 1
    # Append initial dummy label (= uniform distribution)
    first_label = np.array([1] * output_size) / output_size
    true_labels = np.vstack((first_label, true_labels))
    # Make true labels a Variable
    true_labels = Variable(torch.from_numpy(true_labels).float()).view(-1, 1, output_size)

    # Initialize zero vector
    BOD_vector = Variable(torch.zeros((1, 300)))
    # Initialize the label prediction as uniform ditribution over output dimensions
    label_pred = Variable(torch.FloatTensor([1] * output_size) / output_size).view(1, -1)
    # Initialize loss
    loss = Variable(torch.zeros(1))

    if USE_CUDA:
        sentences = sentences.cuda()
        labels = labels.cuda()
        label_pred = label_pred.cuda()
        loss = loss.cuda()
        BOD_vector = BOD_vector.cuda()
        true_labels = true_labels.cuda()

    # Reset gradients
    for optim in optims:
        optim.zero_grad()

    encoder_outputs, encoder_hidden = encoder(sentences)
    decoder_hidden = encoder_hidden

    # For decoding, add the beginning-of-document vector (which is for now just the 0-vector) to the sequence
    sentences = torch.cat((BOD_vector, sentences), 0)

    label_preds = np.zeros((len(encoder_outputs))).astype(int)

    # Now we iterate over the decoder steps
    for i, decoding_step in enumerate(encoder_outputs):
        # Decide whether predicted or true labels are used
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        if use_teacher_forcing:
            output, decoder_hidden = decoder(true_labels[i], sentences[i], decoder_hidden)
        else:
            output, decoder_hidden = decoder(label_pred, sentences[i], decoder_hidden)
        label_pred, output = attn(output, encoder_outputs[i])
        loss += criterion(output, labels[i])
        label_preds[i] = label_pred.max(-1)[1].data[0]

    # Compute mean loss
    loss /= len(encoder_outputs)
    loss.backward()

    # Clip gradients, just to be sure
    clip = 5
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(attn.parameters(), clip)

    # Run optimization step
    for optim in optims:
        optim.step()

    return loss.data[0], label_preds
