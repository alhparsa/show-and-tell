from __future__ import print_function
import torch
from torchvision import datasets, models, transforms
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import torch.nn as nn
from torch import np
import utils
from data_loader import get_coco_data_loader
from models import CNN, RNN
from vocab import Vocabulary, load_vocab
import os

def main(args):
    # hyperparameters
    batch_size = args.batch_size
    num_workers = 1

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    # load COCOs dataset
    IMAGES_PATH = 'data/train2014'
    CAPTION_FILE_PATH = 'data/annotations/captions_train2014.json'

    vocab = load_vocab()
    train_loader = get_coco_data_loader(path=IMAGES_PATH,
                                        json=CAPTION_FILE_PATH,
                                        vocab=vocab,
                                        transform=transform,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)

    # Assumes we extracted Wojtek's devset in data/
    IMAGES_PATH = 'data/data_pack/images/dev2014'
    CAPTION_FILE_PATH = 'data/data_pack/captions_dev2014.json'
    val_loader = get_coco_data_loader(path=IMAGES_PATH,
                                      json=CAPTION_FILE_PATH,
                                      vocab=vocab,
                                      transform=transform,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)


    losses_val = []
    losses_train = []

    # Build the models
    ngpu = 1
    initial_step = 0
    embed_size = 512
    num_hiddens = 512
    learning_rate = 2e-4
    num_epochs = 3
    log_step = 125
    save_step = 1000
    checkpoint_path = 'checkpoints'

    encoder = CNN(embed_size)
    decoder = RNN(embed_size, num_hiddens, len(vocab), 1, rec_unit=args.rec_unit)

    # Loss
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint_file:
        encoder_state_dict, decoder_state_dict, optimizer, initial_step,\
                epoch = utils.load_models(args.checkpoint_file)
        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)
    else:
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.batchnorm.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    if args.sample:
        return utils.sample(encoder, decoder, vocab, val_loader)

    # Train the Models
    total_step = len(train_loader)
    try:
        for epoch in range(num_epochs):
            for step, (images, captions, lengths) in enumerate(train_loader, start=initial_step):

                # Set mini-batch dataset
                images = utils.to_var(images, volatile=True)
                captions = utils.to_var(captions)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                
                # Forward, Backward and Optimize
                decoder.zero_grad()
                encoder.zero_grad()

                if ngpu > 1:
                    # run on multiple GPU
                    features = nn.parallel.data_parallel(encoder, images, range(ngpu))
                    outputs = nn.parallel.data_parallel(decoder, features, range(ngpu))
                else:
                    # run on single GPU
                    features = encoder(images)
                    outputs = decoder(features, captions, lengths)

                train_loss = criterion(outputs, targets)
                losses_train.append(train_loss.data[0])
                train_loss.backward()
                optimizer.step()

                # Run validation set and predict
                if step % log_step == 0:
                    encoder.batchnorm.eval()
                    # run validation set
                    validation_error = 0
                    for val_step, (images, captions, lengths) in enumerate(val_loader):
                        images = utils.to_var(images, volatile=True)
                        captions = utils.to_var(captions, volatile=True)

                        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                        features = encoder(images)
                        outputs = decoder(features, captions, lengths)
                        val_loss = criterion(outputs, targets)
                        validation_error += val_loss.data[0]

                    losses_val.append(validation_error)

                    # predict
                    sampled_ids = decoder.sample(features)
                    sampled_ids = sampled_ids.cpu().data.numpy()[0]
                    sentence = utils.convert_back_to_text(sampled_ids, vocab)
                    print('Sample:', sentence)

                    true_ids = captions.cpu().data.numpy()[0]
                    sentence = utils.convert_back_to_text(true_ids, vocab)
                    print('Target:', sentence)

                    print('Epoch: {} - Step: {} - Train Loss: {} - Eval Loss: {}'.format(epoch, step, train_loss.data[0], val_loss.data[0]))
                    encoder.batchnorm.train()

                # Save the models
                if (step+1) % save_step == 0:
                    utils.save_models(encoder, decoder, optimizer, epoch, step, checkpoint_path)
                    utils.dump_losses(losses_train, losses_val, os.path.join(checkpoint_path, 'losses.pkl'))

    except KeyboardInterrupt:
        utils.dump_losses(losses_train, losses_val, os.path.join(checkpoint_path, 'losses.pkl'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str,
            default=None, help='path to saved checkpoint')
    parser.add_argument('--batch_size', type=int,
            default=128, help='size of batches')
    parser.add_argument('--rec_unit', type=str,
            default='gru', help='choose "gru", "lstm" or "elman"')
    parser.add_argument('--sample', default=False, 
            action='store_true', help='just show result, requires --checkpoint_file')
    args = parser.parse_args()
    main(args)