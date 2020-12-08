from data_loader import get_coco_data_loader, get_basic_loader
from models import CNN_show_attend_tell, RNN_show_attend_tell
from torchvision import datasets, models, transforms
from torch.nn.utils.rnn import pack_padded_sequence
from models import CNN_show_tell, RNN_show_tell
from data_loader import get_coco_data_loader
from vocab import Vocabulary, load_vocab
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import utils
import torch
import tqdm
import os


def main(args):
    """
    Modified version of Muggin's Show and Tell `train.py` function.
    """
    # hyperparameters
    batch_size = args.batch_size
    num_workers = 2

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # load COCOs dataset
    IMAGES_PATH = './data/train2014'
    CAPTION_FILE_PATH = './data/annotations/captions_train2014.json'

    vocab = load_vocab()
    train_loader = get_coco_data_loader(path=IMAGES_PATH,
                                        json=CAPTION_FILE_PATH,
                                        vocab=vocab,
                                        transform=transform,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)

    IMAGES_PATH = './data/val2014'
    CAPTION_FILE_PATH = './data/annotations/captions_val2014.json'
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
    initial_step = initial_epoch = 0
    embed_size = args.embed_size
    num_hiddens = args.num_hidden
    learning_rate = args.lr
    num_epochs = args.epochs
    checkpoint_dir = args.checkpoint_dir
    log_step = args.log_step
    if args.with_attention:
        encoder = CNN_show_attend_tell()
        decoder = RNN_show_attend_tell(attention_dim=args.num_hidden,
                                       embed_dim=args.num_hidden,
                                       decoder_dim=args.num_hidden,
                                       vocab_size=len(vocab))
    else:
        encoder = CNN_show_tell(rcnn=False)
        decoder = RNN_show_tell(embed_size, num_hiddens, len(
            vocab), 5, rec_unit=args.rec_unit)

    # Loss
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint_file:
        _, decoder_state_dict, optimizer, * \
            meta = utils.load_models(args.checkpoint_file)
        initial_step, initial_epoch, losses_train, losses_val = meta
        decoder.load_state_dict(decoder_state_dict)
    else:
        params = list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    counter = 0
    # Train the Models
    total_step = len(train_loader)
    try:
        for epoch in tqdm.notebook.tqdm(range(initial_epoch, num_epochs)):
            tmp_train_loss = []
            for step, (images, captions, lengths) in tqdm.notebook.tqdm(enumerate(train_loader, start=initial_step)):
                counter += 1
                # Set mini-batch dataset
                images = utils.to_var(images, volatile=True)
                captions = utils.to_var(captions)

                # Forward, Backward and Optimize
                optimizer.zero_grad()

                # run on single GPU
                with torch.no_grad():
                    features = encoder(images)
                if args.with_attention:
                    scores, caps_sorted, decode_lengths, alphas = decoder(features, captions, torch.tensor(lengths).cuda())
                    targets = caps_sorted[:, 1:]
                    outputs = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
                else:
                    outputs = decoder(features, captions, lengths)
                    targets = pack_padded_sequence(
                        captions, lengths, batch_first=True)[0]
                train_loss = criterion(outputs, targets)
                losses_train.append(float(train_loss.cpu()))
                tmp_train_loss.append(float(train_loss.cpu()))
                train_loss.backward()
                optimizer.step()
                # Save the models
                if (step+1) % log_step == 0:
                    print('Step: {} - Train Loss: {}'.format(step,
                                                             np.mean(tmp_train_loss)))
                    tmp_train_loss = []

            print('Step: {} - Train Loss: {}'.format(step, np.mean(tmp_train_loss)))
            tmp_train_loss = []
            print('Saving the model')
            utils.save_models(encoder, decoder, optimizer, step,
                              epoch, losses_train, losses_val, checkpoint_dir)
            utils.dump_losses(losses_train, losses_val,
                              os.path.join(checkpoint_dir, 'losses.pkl'))

    except KeyboardInterrupt:
        pass
    finally:
        # Do final save
        utils.save_models(encoder, decoder, optimizer, step,
                          epoch, losses_train, losses_val, checkpoint_dir)
        utils.dump_losses(losses_train, losses_val,
                          os.path.join(checkpoint_dir, 'losses.pkl'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str,
        default=None, help='path to saved checkpoint')
    parser.add_argument('--checkpoint_dir', type=str,
        default='./', help='directory to save checkpoints')
    parser.add_argument('--batch_size', type=int,
        default=256, help='size of batches')
    parser.add_argument('--rec_unit', type=str,
        default='lstm', help='choose "gru", "lstm" or "elman"')
    parser.add_argument('--sample', default=False, 
        action='store_true', help='just show result, requires --checkpoint_file')
    parser.add_argument('--num_hidden', type=int,
        default=512, help='number of hidden units in the RNN')
    parser.add_argument('--embed_size', type=int,
        default=1000, help='number of embeddings in the RNN')
    parser.add_argument('--with_attention', type=bool,
                        default=True, help='Show tell and attend implementation')
    parser.add_argument('--log_step', type=int,
                    default=100, help='Log the training loss after specified steps')
    parser.add_argument('--lr', type=float,
                    default=1e-3, help='learning rate')     
    parser.add_argument('--epochs', type=int,
                    default=20, help='Number of epochs')      
      
    args = parser.parse_args()
    main(args)
