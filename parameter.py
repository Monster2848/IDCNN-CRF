# -!- coding: utf-8 -!-
import os
import pickle


class Parameter():

    def __init__(self):
        self.compound_parameter()
        self.Txt_parameter()
        self.TFrecord_parameter()
        self.Model_parameter()

    def compound_parameter(self):
        self.pattern = 'train'
        self.info_path = os.path.join('data_path', 'info.json')
        self.batch_size = 20
        self.file_generation = False

    def Txt_parameter(self):
        self.data_amount = 4
        self.txt_paths = os.path.join('data_path', 'train_txt')

    def TFrecord_parameter(self):
        self.num_datashards = 1
        self.batch_size_multiplier = 1
        self.capacity = 64 * self.num_datashards
        self.drop_long_sequences = True
        self.is_train = True if self.pattern == 'train' else False
        if self.pattern == 'train':
            self.epoch = 20000
            self.record_path = os.path.join('data_path', 'train_data')
        elif self.pattern == 'test':
            self.epoch = 1
            self.record_path = os.path.join('data_path', 'test_data')

        self.max_length = 3500 or self.batch_size
        self.min_length = 8
        self.mantissa_bits = 2

    def Model_parameter(self):
        self.embedding_path = os.path.join('data_path', 'embeddings.pkl')
        self.embedding_size = 300
        self.original_embedding_size = 300
        with open(self.embedding_path, 'rb') as fb:
            self.embeddings = pickle.load(fb)
        self.filter_width = 3
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.num_filter = 100
        self.lr = 0.001
        self.clip = 5.0
        self.logdir = 'summary'
        self.logger_dir = 'log'
        self.output_path = 'output'
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.exists(self.logger_dir):
            os.makedirs(self.logger_dir)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)