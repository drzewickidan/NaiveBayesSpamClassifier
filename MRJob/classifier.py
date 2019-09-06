import csv
import json
import math
import re
import sys

from collections import Counter
from functools import reduce
from mr3px.csvprotocol import CsvProtocol
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONProtocol
from trainer import TrainingData


WORD_RE = re.compile(r"[\w']+")
OUTPUT_PROTOCOL = CsvProtocol


with open('trained_data.json') as td:
    file_content = re.split(r'[\t\n]', td.read())
if file_content[0] == 'spam':
    spam_data = TrainingData.fromJSON(json.loads(file_content[1]))
    ham_data = TrainingData.fromJSON(json.loads(file_content[3]))
else:
    spam_data = TrainingData.fromJSON(json.loads(file_content[3]))
    ham_data = TrainingData.fromJSON(json.loads(file_content[1]))
p_cat_spam = spam_data.sample_count / (spam_data.sample_count + ham_data.sample_count)
p_cat_ham = ham_data.sample_count / (ham_data.sample_count + spam_data.sample_count)
    
    
def prob(word):
    # use add-one smoothing
    p_word_given_spam = math.log((spam_data.words[word] + 1) / (spam_data.word_count * 2))
    p_word_given_ham = math.log((ham_data.words[word] + 1) / (ham_data.word_count * 2))
    return p_word_given_spam, p_word_given_ham


class MRSpamClassifier(MRJob):

    def steps(self):
        return [MRStep(mapper=self.classifier_mapper,
                       reducer=self.classifier_reducer)]

    def classifier_mapper(self, _, text):
        parsed_text = list(csv.reader([text]))[0][1]
        for word in WORD_RE.findall(parsed_text):
            ps, ph = prob(word)
            yield parsed_text, (ps, ph)
        

    def classifier_reducer(self, text, probabilities):
        p_spam = math.log(p_cat_spam)
        p_ham = math.log(p_cat_ham)
        for p in probabilities:
            p_spam += p[0]
            p_ham += p[1]
        if p_spam > p_ham:
            yield (text, "spam")
        else:
            yield (text, "ham")


if __name__ == '__main__':
    MRSpamClassifier.run()