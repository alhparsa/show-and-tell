from data_loader import get_coco_data_loader, get_basic_loader
from models import CNN_show_attend_tell, RNN_show_attend_tell
from torchvision import datasets, models, transforms
from torch.nn.utils.rnn import pack_padded_sequence
from models import CNN_show_tell, RNN_show_tell
from vocab import Vocabulary, load_vocab
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch import np
import utils
import torch
import os

def main(args):
    """
    Modified version of eval code from muggin's Show and Tell
    """
    # hyperparameters
    batch_size = args.batch_size
    num_workers = 1

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

    vocab = load_vocab()

    if args.with_attention:
        if batch_size > 1:
                print('cannot generate captions, please set the batch size to 1\
                        for attention model.')
                return
    loader = get_basic_loader(dir_path=os.path.join(args.image_path),
                              transform=transform,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)


    # Build the models
    embed_size = args.embed_size
    num_hiddens = args.num_hidden
    checkpoint_path = 'checkpoints'


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

    _, decoder_state_dict, optimizer, *meta = utils.load_models(args.checkpoint_file)
    # encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Train the Models
    try:
        results = []
        for step, (images, image_ids) in enumerate(loader):
            images = utils.to_var(images, volatile=True)

            features = encoder(images)
            if args.with_attention:
                seq = generate_caption(encoder, decoder, images.cuda(),vocab)
                captions = [" ".join([vocab.idx2word[s] for s in seq.detach().cpu().numpy() if s!=1 and s!=2])]
            else:
                captions = decoder.sample(features)
                captions = captions.cpu().data.numpy()
                captions = [utils.convert_back_to_text(cap, vocab) for cap in captions]
            captions_formatted = [{'image_id': int(img_id), 'caption': cap} for img_id, cap in zip(image_ids, captions)]
            results.extend(captions_formatted)
    except KeyboardInterrupt:
        print('Ok bye!')
    finally:
        import json
        file_name = 'captions_model.json'
        with open(file_name, 'w') as f:
            json.dump(results, f)


def generate_caption(encoder, decoder, image, vocabs):
    """
    Modified version of `caption_image_beam_search` from sgrvinod's
    Show Attend and Tell.
    """
    k = 1
    vocab_size = len(vocabs)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)**0.5
    encoder_dim = encoder_out.size(2)

    num_pixels = encoder_out.size(1)
    encoder_out = encoder_out.expand(1, num_pixels, encoder_dim)

    k_prev_words = torch.LongTensor([[vocabs.word2idx['<start>']]] * k).cuda()  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).cuda()  # (k, 1)
    enc_image_size = int(enc_image_size)

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    for i in range(25):
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        top_k_scores, top_k_words = scores.max(1)
        prev_word_inds = torch.tensor([0])  # (s)
        next_word_inds = top_k_words  # (s)
        seqs = torch.cat([seqs[torch.tensor([0])], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        incomplete_inds = [0]
        if next_word_inds in [0,1,2,5]: # Special characters
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
    return seqs[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str,
        default=None, help='path to saved checkpoint')
    parser.add_argument('--batch_size', type=int,
        default=1, help='size of batches')
    parser.add_argument('--rec_unit', type=str,
        default='lstm', help='choose "gru", "lstm" or "elman"')
    parser.add_argument('--image_path', type=str,
        default='data/val2014', help='path to the directory of images')
    parser.add_argument('--num_hidden', type=int,
        default=512, help='number of hidden units in the RNN')
    parser.add_argument('--embed_size', type=int,
        default=1000, help='number of embeddings in the RNN')
    parser.add_argument('--with_attention', type=bool,
        default=True, help='Show attend and tell implementation')
    args = parser.parse_args()
    main(args)
