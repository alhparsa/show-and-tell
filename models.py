import torch

from torch import nn
from torch.nn import Sequential
from torchvision import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models.detection import FasterRCNN


class Attention(nn.Module):
    """
    Based on https://github.com/AaronCCWong/Show-Attend-and-Tell
    """

    def __init__(self, encoder_dim, hidden_size):
        super(Attention, self).__init__()
        self.U = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(encoder_dim, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha


class CNN_show_attend_tell(nn.Module):
    """
    Based on https://github.com/AaronCCWong/Show-Attend-and-Tell
    """

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


class RNN_show_attend_tell(nn.Module):
    """
    Based on https://github.com/AaronCCWong/Show-Attend-and-Tell
    """

    def __init__(self, vocabulary_size, hidden_size, encoder_dim):
        super(RNN_show_attend_tell, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(hidden_size, vocabulary_size)
        self.dropout = nn.Dropout()

        self.attention = Attention(encoder_dim, hidden_size)
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size + encoder_dim, hidden_size)

    def forward(self, img_features, captions):
        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1

        prev_words = torch.zeros(batch_size, 1).long().cuda()
        embedding = self.embedding(prev_words)

        preds = torch.zeros(batch_size, max_timespan,
                            self.vocabulary_size).cuda()
        alphas = torch.zeros(batch_size, max_timespan,
                             img_features.size(1)).cuda()
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            embedding = embedding.squeeze(
                1) if embedding.dim() == 3 else embedding
            lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training:
                embedding = self.embedding(
                    output.max(1)[1].reshape(batch_size, 1))
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c

    def caption(self, img_features, beam_size=1, max_len=25):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        prev_words = torch.zeros(beam_size, 1).long()

        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, img_features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(
                    -1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat(
                (sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat(
                (alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(
                next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > max_len:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha


class CNN_show_tell(nn.Module):
    """Class to build new model including all but last layers"""

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
