import torch

from torch import nn
from torch.nn import Sequential
from torchvision import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models.detection import FasterRCNN


class CNN(nn.Module):
    """Class to build new model including all but last layers"""

    def __init__(self, output_dim=1000):
        super(CNN, self).__init__()
        self.pre_trained = models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        self.linear = nn.Linear(91, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        self.pre_trained.eval()
        x_ = self.pre_trained(x)
        s = torch.zeros(x.shape[0], 91)
        for b in range(x.shape[0]):
            labels, scores = x_[b]['labels'], x_[b]['scores']
            for i, l in enumerate(labels):
                s[b][l] += float(scores[i])
        out = torch.relu(self.batchnorm(self.linear(s.cuda())))
        return out


class RNN(torch.nn.Module):
    """
    Recurrent Neural Network for Text Generation.
    To be used as part of an Encoder-Decoder network for Image Captioning.
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
        assert rec_unit in RNN.__rec_units, 'Specified recurrent unit is not available'

        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, 1000)
        self.unit = RNN.__rec_units[rec_unit](1000, hidden_size, num_layers,
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
