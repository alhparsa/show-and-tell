from torchvision.models.detection import FasterRCNN
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn import Sequential
from torchvision import models
from torch import nn
import torch

class CNN_show_attend_tell(nn.Module):
    def __init__(self):
        super(CNN_show_attend_tell, self).__init__()
        self.pre_trained = models.resnet152(True)
        self.pre_trained = nn.Sequential(
            *list(self.pre_trained.children())[:-2])
        for param in self.pre_trained.parameters():
            param.requires_grad = False
        self.pre_trained.eval()

    def forward(self, x):
        out = self.pre_trained(x)
        out = out.permute(0, 2, 3, 1)
        out = out.view(out.size(0), -1, out.size(-1))
        return out

class Attention(nn.Module):
    """
    Attention Network. Based on sgrvinod's Show Attend and Tell.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha




class RNN_show_attend_tell(nn.Module):
    """
    Decoder. Based on sgrvinod's Show Attend and Tell.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(RNN_show_attend_tell, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda()
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).cuda()

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(h)  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas




class CNN_show_tell(nn.Module):

    def __init__(self, output_dim=1000, rcnn=True):
        super(CNN_show_tell, self).__init__()
        if rcnn:
            self.pre_trained = models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True)
        else:
            self.pre_trained = models.resnet152(True)
        for param in self.pre_trained.parameters():
            param.requires_grad = False
        self.rcnn = rcnn
        self.pre_trained.eval()
        if rcnn:
            self.linear = nn.Linear(91, output_dim)
            self.batchnorm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        with torch.no_grad():
            out = self.pre_trained(x)
        if self.rcnn:
            s = torch.zeros(x.shape[0], 91)
            for b in range(x.shape[0]):
                labels, scores = out[b]['labels'], out[b]['scores']
                for i, l in enumerate(labels):
                    s[b][l] += float(scores[i])
            out = torch.relu(self.batchnorm(self.linear(out.cuda())))
        return out


class RNN_show_tell(torch.nn.Module):
    """
    Based on muggin's Show and Tell model.
    """
    __rec_units = {
        'elman': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, emb_size, hidden_size, vocab_size, num_layers=1, rec_unit='gru'):
        """
        Initializer

        :param embed_size: size of word embeddings
        :param hidden_size: size of hidden state of the recurrent unit
        :param vocab_size: size of the vocabulary (output of the network)
        :param num_layers: number of recurrent layers (default=1)
        :param rec_unit: type of recurrent unit (default=gru)
        """
        rec_unit = rec_unit.lower()
        assert rec_unit in RNN_show_tell.__rec_units, 'Specified recurrent unit is not available'

        super(RNN_show_tell, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.unit = RNN_show_tell.__rec_units[rec_unit](emb_size, hidden_size, num_layers,
                                                        batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        """
        Forward pass through the network

        :param features: features from CNN feature extractor
        :param captions: encoded and padded (target) image captions
        :param lengths: actual lengths of image captions
        :returns: predicted distributions over the vocabulary
        """
        # embed tokens in vector space
        embeddings = self.embeddings(captions)

        # append image as first input
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)

        # pack data (prepare it for pytorch model)
        inputs_packed = pack_padded_sequence(inputs, lengths, batch_first=True)

        # run data through recurrent network
        hiddens, _ = self.unit(inputs_packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, max_len=25):
        """
        Sample from Recurrent network using greedy decoding

        :param features: features from CNN feature extractor
        :returns: predicted image captions
        """
        output_ids = []
        states = None
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            # pass data through recurrent network
            hiddens, states = self.unit(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # find maximal predictions
            predicted = outputs.max(1)[1]

            # append results from given step to global results
            output_ids.append(predicted)

            # prepare chosen words for next decoding step
            inputs = self.embeddings(predicted)
            inputs = inputs.unsqueeze(1)
        output_ids = torch.stack(output_ids, 1)
        return output_ids.squeeze()
