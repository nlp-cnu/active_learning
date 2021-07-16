import numpy as np
import pandas
from transformers import AutoTokenizer
import preprocessor as p

df = pandas.read_csv('full_dataset.tsv', header=None, names=['tweet', 'label'], delimiter='\t').dropna()
data = df['tweet']

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

avg = np.average(num_tokens)
std = np.std(num_tokens)

print(f'Average tokens: {avg}')
print(f'Std Dev: {std}')
print(f'avg + (2 * std) = {avg + (2 * std)}')
print(f"Max tokens: {np.max(num_tokens)}")
print(len([x for x in num_tokens if x > 64]))

# justify reductions to max tokens: Due to memory constrains and training speed,
# reduces max tokens to avg + 2 std dev, cuts off 2.2% of total samples
