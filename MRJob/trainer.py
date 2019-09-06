import csv
import json
import re
import sys

from collections import Counter
from functools import reduce
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONProtocol


WORD_RE = re.compile(r"[\w']+")


class TrainingData():

    def __init__(self):
        self.words = Counter()
        self.word_count = 0
        self.sample_count = 0

    @staticmethod
    def combine(td1, td2):
        td = TrainingData()
        td.word_count = td1.word_count + td2.word_count
        td.sample_count = td1.sample_count + td2.sample_count
        td.words = td1.words + td2.words
        return td

    def toJSON(self):
        return self.__dict__
        
    @staticmethod
    def fromJSON(obj):
        # handle the change in type during reduce
        if type(obj) is TrainingData:
            return obj
        ret = TrainingData()
        ret.words = Counter(obj['words'])
        ret.word_count = obj['word_count']
        ret.sample_count = obj['sample_count']
        return ret
    

class MRSpamTrainer(MRJob):

    def steps(self):
        return [MRStep(mapper=self.trainer_mapper,
                       reducer=self.trainer_reducer,
                       combiner=self.trainer_reducer)]

    def trainer_mapper(self, _, line):
        mapped_data = {
            'spam': TrainingData(),
            'ham': TrainingData()
        }
        parsed_line = list(csv.reader([line]))[0]
        spam_ham = parsed_line[0]
        text = parsed_line[1]

        mapped_data[spam_ham].sample_count = 1

        for word in WORD_RE.findall(text):
            lword = word.lower()
            if lword not in mapped_data[spam_ham].words:
                mapped_data[spam_ham].words[lword] = 0
            mapped_data[spam_ham].words[lword] += 1
            mapped_data[spam_ham].word_count += 1

        yield 'spam', mapped_data['spam'].toJSON()
        yield 'ham', mapped_data['ham'].toJSON()

    def trainer_reducer(self, spam_ham, training_data):
        trained_data = reduce(
            lambda a, b: TrainingData.combine(
                TrainingData.fromJSON(a),
                TrainingData.fromJSON(b)),
            training_data)
        yield spam_ham, trained_data.toJSON()


if __name__ == '__main__':
    MRSpamTrainer.run()