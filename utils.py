# -!- coding: utf-8 -!-
import json
import os
import time
import numpy as np
import tensorflow as tf
from parameter import Parameter


class Txt(Parameter):

    def __init__(self):
        Parameter.__init__(self)
        self.info_json()

    def info_json(self):
        with open(self.info_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.index2label = {v_: k_ for k_, v_ in data['tag_id'].items()}
        self.word2index = data['word2index']
        if 'tag_id' in data:
            self.tag_id = data['tag_id']
        else:
            self.tag_id = {"<PAD>": 0, 'O': 1}

    def txt_path(self):
        subscript = self.data_amount
        paths = [list() for _ in range(self.data_amount)]
        for json_path in os.listdir(self.txt_paths):
            if subscript:
                subscript -= 1
                paths[subscript].append(os.path.join(self.txt_paths, json_path))
            else:
                paths[subscript].append(os.path.join(self.txt_paths, json_path))
                subscript = self.data_amount
        return paths

    def txt_load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            text_data, label = zip(*[x.split('\t') for x in f.read().split('\n') if '\t' in x])

        return text_data, label

    def labels_output(self, global_step, targets, decodes, texts, ids):
        output_path = os.path.join(self.output_path, self.pattern)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if len(os.listdir(output_path)) and self.pattern == 'train':
            os.remove(os.path.join(output_path, os.listdir(output_path)[0]))
        output_path = os.path.join(output_path, str(global_step) + '.json')
        data = {
            id.decode('utf-8'): {
                'labels_true': self.output_utils([self.index2label[x] for x in target], text),
                'labels_pred': self.output_utils([self.index2label[x] for x in decode], text),
            }
            for target, decode, text, id in zip(targets, decodes, texts, ids)}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def output_utils(self, entrys, texts):
        texts = texts if isinstance(texts,str) else texts.decode('utf-8')
        data = {}
        tags = []
        _ = []
        for entry, text in zip(entrys, texts):
            if entry == "O":
                if len(_) > 0:
                    tags.append(_)
                _ = []
            elif 'B' in entry:
                if len(_) > 0:
                    tags.append(_)
                _ = []
                _.append((entry, text))
            else:
                _.append((entry, text))
        if len(_) > 0:
            tags.append(_)
        for tag in tags:
            for k, v in tag:
                if k[2:] in data:
                    if _ == k[2:]:
                        data[k[2:]].append(v)
                        _ = ''
                    else:
                        data[k[2:]][len(data[k[2:]]) - 1] += v
                else:
                    data[k[2:]] = [v]
            _ = k[2:]
        return data


class TFrecord(Txt):

    def record_dict(self, id, text, words, label):
        return {
            'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf_8')])),
            'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode('utf_8')])),
            'char_inputs': tf.train.Feature(int64_list=tf.train.Int64List(value=words)),
            'targets': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
        }

    def __key_features(self):
        return {
            'id': tf.FixedLenFeature([1], tf.string),
            'text': tf.FixedLenFeature([1], tf.string),
            'char_inputs': tf.VarLenFeature(tf.int64),
            'targets': tf.VarLenFeature(tf.int64)
        }

    def __items_handlers(self):
        return {
            "id": tf.contrib.slim.tfexample_decoder.Tensor("id"),
            "text": tf.contrib.slim.tfexample_decoder.Tensor("text"),
            "char_inputs": tf.contrib.slim.tfexample_decoder.Tensor("char_inputs"),
            "targets": tf.contrib.slim.tfexample_decoder.Tensor("targets")
        }

    def record_load(self):
        feature_map = self.record_batch()
        return (
            tf.squeeze(feature_map['id'], axis=-1),
            tf.squeeze(feature_map['text'], axis=-1),
            tf.cast(feature_map['char_inputs'], tf.int32, name="ChatInputs"),
            tf.cast(feature_map['targets'], tf.int32, name="Targets")
        )

    def record_queue(self):
        with tf.name_scope("examples_queue"):
            num_readers = min(4 if self.is_train else 1, len(
                tf.contrib.slim.parallel_reader.get_data_files(os.path.join(self.record_path, '*.tfrecord'))))
            _, examples = tf.contrib.slim.parallel_reader.parallel_read(
                data_sources=[os.path.join(self.record_path, '*.tfrecord')], reader_class=tf.TFRecordReader,
                num_epochs=self.epoch if self.is_train else 1, shuffle=self.is_train, capacity=2 * self.capacity,
                min_after_dequeue=self.capacity, num_readers=num_readers)
            decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(self.__key_features(), self.__items_handlers())
            decoded = decoder.decode(examples, items=list(self.__items_handlers()))
            return {field: tensor for (field, tensor) in zip(self.__key_features(), decoded)}

    def record_batch(self):
        with tf.name_scope("batch_examples"):
            x = self.min_length
            boundaries = []
            while x < self.max_length:
                boundaries.append(x)
                x += 2 ** max(0, int(np.log2(x)) - self.mantissa_bits)

            batch_sizes = [self.batch_size for _ in range(len(boundaries) + 1)]

            max_example_length = 0
            for v in self.record_queue().values():
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

            (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
                max_example_length,
                self.record_queue(),
                batch_sizes,
                [b + 1 for b in boundaries],
                dynamic_pad=True,
                keep_input=(max_example_length <= self.max_length)
            )
            return outputs

    def record_write(self):
        for i, paths in enumerate(self.txt_path()):
            if not os.path.exists(self.record_path):
                os.makedirs(self.record_path)
            tagger_writer = tf.python_io.TFRecordWriter(
                os.path.join(self.record_path, '{}_'.format(self.pattern) + str(i) + '.tfrecord')
            )
            for path in paths:
                try:
                    print(path)
                    content, labels = self.txt_load(path)
                    words = [self.word2index[c] if c in self.word2index else self.word2index['<UNK>'] for c in
                             content]
                    label = list()
                    for c in labels:
                        if c not in self.tag_id:
                            self.tag_id[c] = len(self.tag_id)
                        label.append(self.tag_id[c])

                    tagger_writer.write(tf.train.Example(
                        features=tf.train.Features(feature=self.record_dict(path, ''.join(content), words, label))
                    ).SerializeToString())
                except TypeError:
                    pass

            tagger_writer.close()
        self.info_update()

    def info_update(self):
        data = {
            'word2index': self.word2index,
            'tag_id': self.tag_id
        }
        with open(self.info_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def run(self):
        self.record_write()
