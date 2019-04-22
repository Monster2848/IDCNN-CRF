# -!- coding: utf-8 -!-
import logging
import os
import tensorflow as tf
from utils import TFrecord


class Crf_model(TFrecord):
    def __init__(self):
        TFrecord.__init__(self)
        if self.file_generation:
            self.run()
        self.__model_main()
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.INFO)
        # Formator
        formator = logging.Formatter(
            fmt='%(asctime)s %(levelname)s in %(funcName)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(
                self.logger_dir,
                os.path.basename(self.logdir).title() + '.log'
            ),
            encoding='utf_8'
        )
        file_handler.setFormatter(formator)

    def __placeholder(self):
        self.__global_step = tf.train.get_or_create_global_step()
        if self.pattern == 'train':
            self.id, self.text, self.char_inputs, self.targets = self.record_load()
            self.dropout = tf.constant(0.5)
        elif self.pattern == 'test':
            self.batch_size = 1
            self.id, self.text, self.char_inputs, self.targets = self.record_load()
            self.dropout = tf.constant(1.0)
        else:
            self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ChatInputs")
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
            self.dropout = tf.constant(1.0)
        __used = tf.sign(tf.abs(self.char_inputs))
        __length = tf.reduce_sum(__used, reduction_indices=1)
        self.lengths = tf.cast(__length, tf.int32)
        self.num_steps = tf.shape(self.char_inputs)[-1]

    def __word_embedding(self):
        with tf.name_scope('embedding'):
            word_embeddings_big = tf.Variable(
                self.embeddings, dtype=tf.float32, name="word_embeddings_big",
                trainable=False
            )
            word_embeddings_w = tf.get_variable(
                'word_embeddings_w',
                shape=[self.original_embedding_size, self.embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32,
                trainable=True
            )
            word_embeddings = tf.matmul(
                word_embeddings_big, word_embeddings_w,
                name='word_embeddings'
            )
            inputs = tf.nn.embedding_lookup(word_embeddings, self.char_inputs)
            self.word_embeddings = tf.nn.dropout(inputs, self.dropout)

    def __idcnn(self):
        model_inputs = tf.expand_dims(self.word_embeddings, 1)
        with tf.variable_scope("idcnn"):
            shape = [1, self.filter_width, self.embedding_size, self.num_filter]
            filter_weights = tf.get_variable("idcnn_filter", shape=shape,
                                             initializer=tf.contrib.layers.xavier_initializer())
            layerInput = tf.nn.conv2d(model_inputs, filter_weights, strides=[1, 1, 1, 1], padding="SAME",
                                      name="init_layer", use_cudnn_on_gpu=True)
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            reuse = False
            if self.dropout == 1.0:
                reuse = True
            for j in range(4):
                for i in range(len(self.layers)):
                    # 1,1,2
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i, reuse=True if (reuse or j > 0) else False):
                        w = tf.get_variable("filterW", shape=[1, self.filter_width, self.num_filter, self.num_filter],
                                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput, w, rate=dilation, padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])

            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[totalWidthForLastDim, len(self.index2label)], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[len(self.index2label)]))

                pred = tf.nn.xw_plus_b(finalOut, W, b)
            self.logits = tf.reshape(pred, [-1, self.num_steps, len(self.index2label)])

    def __loss_layer(self):
        with tf.name_scope('loss'):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                                       tag_indices=self.targets,
                                                                                       sequence_lengths=self.lengths + 1)
            self.loss = tf.abs(tf.reduce_mean(log_likelihood))

            self.log_tensors = {
                "global_step": self.__global_step,
                "inputs": tf.shape(self.char_inputs),
                "labels": tf.shape(self.targets),
                "loss": self.loss
            }

    def __Optimizer(self):
        with tf.variable_scope("optimizer"):
            self.opt = tf.train.AdamOptimizer(self.lr)
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.clip, self.clip), v] for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.__global_step)

    def __model_main(self):
        self.__placeholder()
        self.__word_embedding()
        self.__idcnn()
        self.__loss_layer()
        self.__Optimizer()

    def __transition(self, lengths, logits, transition_params):
        label_list = []
        for logit, seq_len in zip(logits, lengths):
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list

    def train(self):
        hooks = [
            tf.train.LoggingTensorHook(
                self.log_tensors, every_n_iter=1
            ),
            tf.train.CheckpointSaverHook(
                checkpoint_dir=self.logdir, save_secs=20
            ),
            tf.train.SummarySaverHook(
                output_dir=self.logdir, save_steps=10,
                summary_op=tf.summary.merge_all()
            )
        ]
        saver = tf.train.Saver()
        with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=self.logdir) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.logdir))
            if self.pattern == 'train':
                while not sess.should_stop():
                    global_step, logits, ids, texts, targets, lengths, loss, _, transition = sess.run(
                        [self.__global_step, self.logits, self.id, self.text, self.targets, self.lengths, self.loss,
                         self.train_op, self.transition_params])
                    if global_step % 100 == 0:
                        decodes = self.__transition(lengths, logits, transition)
                        self.labels_output(global_step, targets, decodes, texts, ids)

    def demo(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.logdir))
            if self.pattern == 'test':
                global_step = 0
                while not sess.should_stop():
                    logits, ids, texts, targets, lengths, transition = sess.run(
                        [self.logits, self.id, self.text, self.targets, self.lengths, self.transition_params])
                    decodes = self.__transition(lengths, logits, transition)
                    self.labels_output(global_step, targets, decodes, texts, ids)
                    global_step += 1
            else:
                while True:
                    target = input('输入测试的文字:')
                    content = [
                        [self.word2index[c] if c in self.word2index else self.word2index['<UNK>'] for c in target]]
                    logits, loss, lengths, transition = sess.run(
                        [self.logits, self.loss, self.lengths, self.transition_params], feed_dict={
                            self.char_inputs: content,
                            self.targets: [[0] * len(c) for c in content]
                        })
                    decodes = self.__transition(lengths, logits, transition)
                    decodes = [[self.index2label[y] for y in x] for x in decodes]
                    print([self.output_utils(x, target) for x in decodes])

