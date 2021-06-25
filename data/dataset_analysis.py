num_positive = 0
num_negative = 0
# with open('full_dataset.tsv.tsv', encoding='utf8') as f:
#     for line in f:
#         # print(line)
#         label = line.rstrip().split('\t')[1]
#
#         if label == '1':
#             # print(line)
#             num_positive += 1
#         else:
#             num_negative += 1

# print(f'Positive Samples: {num_positive}')
# print(f'Negative Samples: {num_negative}')

# training_samples = 0
# test_samples = 0

file_1_samples = 0
with open('training_set_1_ids.txt', encoding='utf8') as f:
    for line in f:
        file_1_samples += 1

# with open('train.tsv', encoding='utf8') as f:
#     for line in f:
#         training_samples += 1

# with open('validation.tsv', encoding='utf8') as f:
#     for line in f:
#         training_samples += 1

# with open('test.tsv', encoding='utf8') as f:
#     for line in f:
#         test_samples += 1

file_2_samples = 0
with open('training_set_2_ids.txt', encoding='utf8') as f:
    for line in f:
        file_2_samples += 1

file_3_samples = 0
with open('evaluation_set_ids.txt', encoding='utf8') as f:
    for line in f:
        file_3_samples += 1

print(file_1_samples)
print(file_2_samples)
print(file_3_samples)
print(f'{num_negative + num_positive} retrieved out of {file_1_samples + file_2_samples + file_3_samples}')