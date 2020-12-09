import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
#text  = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
#text = "[CLS] Should I help people ? [SEP] The answer is no [SEP]"
text = "[CLS] I am a bad person . [SEP] I like to smile [SEP]"

tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

masked_index = 11 # 10 for kill people
tokenized_text[masked_index] = '[MASK]'
print(tokenized_text)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens, len(indexed_tokens))

segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # kill people [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
assert len(indexed_tokens) == len(segments_ids)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer       of the Bert model
    encoded_layers = outputs[0]


# We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)


# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

topk = 20
selected_topk = 0
_, predicted_indices = torch.topk(predictions[0, masked_index], topk)
predicted_index = predicted_indices[selected_topk].item()
#predicted_index = torch.argmax(predictions[0, masked_index]).item()

predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print("Top answer is", predicted_index,predicted_token)

for idx, predicted_index in enumerate(predicted_indices):
       predicted_index = predicted_index.item()
       predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
       print("Top-{} answer is".format(idx), predicted_index,predicted_token)