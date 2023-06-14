import nltk
import os
import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def read_files(path):
    data = []
    for author_folder in os.listdir(path):
        author_folder_path = os.path.join(path, author_folder)
        if os.path.isdir(author_folder_path):
            label = author_folder
            for file in os.listdir(author_folder_path):
                if file.endswith(".txt"):
                    with open(os.path.join(author_folder_path, file), "r", encoding="utf8") as f:
                        article = f.read()
                    data.extend([(article, label)])
    df = pd.DataFrame(data, columns=["text", "author"])
    return df


current_path = os.getcwd()
df = read_files('/content/drive/MyDrive/VeriMaden/authors')


def print_dataframe(df):
    data = []
    for index, row in df.iterrows():
        words = nltk.word_tokenize(row['text'])
        uniquewordscount = len(set(words))
        tagged_words = nltk.pos_tag(words)
        sentencecount = len(nltk.sent_tokenize(row['text']))
        wordcount = len([word for word in words if word not in string.punctuation])
        prepositioncount = len([word for word, tag in tagged_words if tag.startswith('IN')])
        conjunctioncount = len([word for word, tag in tagged_words if tag.startswith('CC')])
        adjectivecount = len([word for word, tag in tagged_words if tag.startswith('JJ')])
        nouncount = len([word for word, tag in tagged_words if tag.startswith('NN')])
        pronouncount = len([word for word, tag in tagged_words if tag.startswith('PRP')])
        adverbcount = len([word for word, tag in tagged_words if tag.startswith('RB')])
        verbcount = len([word for word, tag in tagged_words if tag.startswith('VB')])
        punctuationcount = len([word for word in words if word in string.punctuation])
        numbercount = len([char for char in row['text'] if char.isdigit()])
        timewordcount = len([word for word in words if
                             word in ['today', 'yesterday', 'tomorrow', 'now', 'tonight', 'morning', 'afternoon',
                                      'evening', 'night']])
        questioncount = len([word for word in words if
                             word in ['where', "which", 'what', 'when', 'who', 'whom', 'whose', 'why', 'how']])
        avg_sentence_length = wordcount / sentencecount
        data.extend([(prepositioncount, conjunctioncount, adjectivecount, nouncount, pronouncount, adverbcount,
                      verbcount, sentencecount, wordcount, uniquewordscount, int(avg_sentence_length), punctuationcount,
                      numbercount, timewordcount, questioncount, str(int(row['author'][:2]) - 1))])
    df = pd.DataFrame(data,
                      columns=["prepositions", "conjunctions", "adjectives", "nouns", "pronouns", "adverbs", "verbs",
                               "sentences", "words", "unique_words", "avg_sentence_length", "punctuations", "numbers",
                               "timewords", "questionwords", "author"])
    return df


df = print_dataframe(df)
df.to_csv('/content/drive/MyDrive/VeriMaden/out.csv', index=False)