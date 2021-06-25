import math
import os
import time

import dotenv
import tweepy
from tqdm import tqdm

# dataset can be downloaded from https://data.mendeley.com/datasets/rxwfb3tysd/2
# download the subtask1.zip file to get the correct files.

# todo: create .env file with your credentials with format
# consumer_key=XXXX
# consumer_key_secret=XXXX
# access_token=XXXX
# access_token_secret=XXXX

# or just copy paste them into the variables

# Get credentials from .env file
env = dotenv.dotenv_values(os.path.join('..', '.env'))

consumer_key = env.get('consumer_key')
consumer_key_secret = env.get('consumer_key_secret')
access_token = env.get('access_token')
access_token_secret = env.get('access_token_secret')

# initialize Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def create_data_file(input_filepath, output_filepath):
    input_file = open(input_filepath, encoding='utf8')
    output_file = open(output_filepath, 'a', encoding='utf8')
    lines = input_file.readlines()

    start_time = time.time()
    iterator = tqdm(lines, desc='Gathering Tweets', total=len(lines))
    for line in iterator:

        # todo format for your input data file
        tweet_id, user_id, label = line.rstrip().split('\t')

        try:
            tweet = api.get_status(tweet_id)
            text = tweet.text.replace('\n', ' ')
            # todo format the way you like
            output_file.write(f'{text}\t{label}\n')

        except tweepy.RateLimitError:
            # rate limit reached. 900 api requests / 15 minute interval

            # todo: you can remove the busy wait by just using the commented out code below,
            #  but the busy wait gives a fun time printout in tqdm
            # time.sleep(wait_time)

            wait_time = (15 * 60) - (time.time() - start_time)
            while wait_time > 0:
                time_string = f'{math.ceil(wait_time / 60)} minute(s)' if wait_time > 1 else f'{wait_time} seconds(s)'
                iterator.set_postfix_str('Rate limited: waiting for ' + time_string)
                time.sleep(1)
                wait_time = (15 * 60) - (time.time() - start_time)

            start_time = time.time()
            iterator.set_postfix_str('')

        except tweepy.TweepError:
            # Tweet unreachable
            pass

    output_file.close()
    input_file.close()


if __name__ == '__main__':
    input_filepath = 'training_set_1_ids.txt'
    output_filepath = 'train.tsv'
    create_data_file(input_filepath, output_filepath)

    input_filepath = 'training_set_2_ids.txt'
    output_filepath = 'validation.tsv'
    create_data_file(input_filepath, output_filepath)

    input_filepath = 'evaluation_set_ids.txt'
    output_filepath = 'test.tsv'
    create_data_file(input_filepath, output_filepath)

    with open('full_dataset.tsv', 'w', encoding='utf8') as f:
        for line in open('train.tsv', encoding='utf8'):
            f.write(line)
        for line in open('validation.tsv', encoding='utf8'):
            f.write(line)
        for line in open('test.tsv', encoding='utf8'):
            f.write(line)
