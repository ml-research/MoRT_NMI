import torch
import torch.nn as nn
from torch import cuda
from sentence_transformers import SentenceTransformer

torch.manual_seed(0)
cuda.manual_seed_all(0)


class Encoder(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout_rate):
        super(Encoder, self).__init__()
        self.hidden_layer = nn.Linear(emb_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.tanh = nn.Tanh()

    def forward(self, emb):
        hidden = self.tanh(self.hidden_layer(self.dropout(emb)))

        return hidden


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden):
        output = self.output_layer(hidden)
        pre = self.sigmoid(output)

        return pre


class Decoder(nn.Module):
    def __init__(self, hidden_size, emb_size, dropout_rate):
        super(Decoder, self).__init__()
        self.output_layer = nn.Linear(hidden_size, emb_size)
        self.tanh = nn.Tanh()

    def forward(self, hidden):
        pre = self.tanh(self.output_layer(hidden))

        return pre


class DictWrapper(nn.Module):
    def __init__(self, module, key_input, key_output):
        super(DictWrapper, self).__init__()
        self.module = module
        self.key_input = key_input
        self.key_output = key_output

    def forward(self, features):
        input_vector = features[self.key_input]
        output_vector = self.module(input_vector)
        features.update({self.key_output: output_vector})
        return features


class SentenceTransformerAE(SentenceTransformer):
    def __init__(self, encoder, decoder, model_name_or_path=None, modules=None, device=None):
        super(SentenceTransformerAE, self).__init__(model_name_or_path, modules, device)
        self.encoder = DictWrapper(encoder, 'sentence_embedding', 'latent_sentence_embedding')
        self.decoder = DictWrapper(decoder, 'latent_sentence_embedding', 'sentence_embedding')
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def forward(self, features):
        features = super(SentenceTransformerAE, self).forward(features)
        #features = self.encoder(features)
        #features = self.decoder(features)
        return features


def init_SentenceTransformerAE(eval_model_path, device="cuda"):
    checkpoint = torch.load(eval_model_path, map_location=lambda storage, loc: storage.cuda(0))
    hp_loaded_model = checkpoint['hp']

    encoder = Encoder(hp_loaded_model.emb_size, hp_loaded_model.hidden_size, dropout_rate=0)
    decoder = Decoder(hp_loaded_model.hidden_size, hp_loaded_model.emb_size, dropout_rate=0)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    # fix random
    model = SentenceTransformerAE(encoder=encoder, decoder=decoder,
                                  model_name_or_path=hp_loaded_model.model_name,
                                  device=device)

    return model, checkpoint['norm_score'] if 'norm_score' in list(checkpoint.keys()) else None
