import numpy as np
import pandas
from transformers import AutoTokenizer
import preprocessor as p

df = pandas.read_csv('full_dataset.tsv', header=None, names=['tweet', 'label'], delimiter='\t').dropna()
data = df['tweet']
#
# y = df['label']
#
# pos = 0
# neg = 0
# for idx in range(len(y)):
#     label = y.iloc[idx]
#     if label == 1:
#         pos += 1
#     else:
#         neg += 1
#
# print(pos)
# print(neg)
# print(neg / pos)

# budget = 50
# df = pandas.read_csv(f'..\\active_learning_scores\\DAL_{budget}_selected_samples.tsv', header=None, names=['tweet', 'label'], delimiter='\t').dropna()
# y = df['label']
#
# pos = 100
# neg = 0
# for num_samples in range(0, len(y), budget):
#     for idx in range(num_samples, min(num_samples + budget, len(y))):
#         label = y.iloc[idx]
#         if label == 1:
#             pos += 1
#         else:
#             neg += 1
#
#     print(f'{num_samples + budget} samples')
#     print(f'Neg / Pos Ratio - {neg // pos} : 1')
#     print()

# preprocess tweets to remove mentions, URL's
p.set_options(p.OPT.MENTION, p.OPT.URL)  # p.OPT.HASHTAG, p.OPT.EMOJI
data = data.apply(p.clean)

# Tokenize special Tweet characters
# p.set_options()  # p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.RESERVED, p.OPT.NUMBER, p.OPT.HASHTAG
# data = data.apply(p.tokenize)

tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base')
tokenized = tokenizer(list(data))

num_tokens = []
for sample in tokenized['input_ids']:
    num_tokens.append(len(sample))

avg = np.ceil(np.average(num_tokens))
std = np.ceil(np.std(num_tokens))
target = avg + (2 * std)
num_cut_off = len([x for x in num_tokens if x > avg + (2 * std)])

print(f'Average tokens:     {avg}')
print(f'Std Dev:            {std}')
print(f'avg + (2 * std):    {target}')
print(f"Max tokens:         {np.max(num_tokens)}")
print(f'Number greater than {target}: {num_cut_off} / {len(num_tokens)} - {num_cut_off / len(num_tokens):%}')

print(len(num_tokens))
print(data.iloc[np.argmax(num_tokens)])

# justify reductions to max tokens: Due to memory constrains and training speed,
# reduces max tokens to avg + 2 std dev, cuts off 2.2% of total samples
