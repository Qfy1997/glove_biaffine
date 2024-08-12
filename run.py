import tensorflow as tf
from collections import defaultdict
import os
import numpy as np
import time

class Token(object):
    def __init__(self, token_id, word, pos, rel=-1, head_id=-1):
        self.token_id = token_id
        self.word = word
        self.pos = pos
        self.rel = rel
        self.head_id = head_id


def load(path, max_len=128):
        dataset = []
        root_token = Token(0, '<root>', '<unk>', '<null>', 0)
        with open(path, encoding='utf8') as f:
            sentence = [root_token]
            for line in f.readlines():
                if line == '\n' or line.startswith('#'):
                    if 1 < len(sentence) <= max_len:
                        dataset.append(sentence)
                    sentence = [root_token]
                    continue
                line = line.split('\t')
                token = Token(int(line[0]), line[1], line[3], line[7], int(line[6]))
                sentence.append(token)
        return dataset


def pad_sequences(sequences, max_len=None,value=0):
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences, max_len, padding='post', truncating='post', value=value)



class MLP(tf.keras.layers.Layer):
    def __init__(self, units, num_layers=1, dropout_rate=0.):
        super(MLP, self).__init__()
        self.denses = []
        for _ in range(num_layers):
            self.denses.append(tf.keras.layers.Dropout(dropout_rate))
            self.denses.append(tf.keras.layers.Dense(units))
    
    def call(self, x, training=False):
        for layer in self.denses:
            x = layer(x, training=training)
        return x


class Biaffine(tf.keras.layers.Layer):
    def __init__(self, in_size, out_size, bias_x=False, bias_y=False):
        super(Biaffine, self).__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.w = self.add_weight(
            name='weight', 
            shape=(out_size, in_size + int(bias_x), in_size + int(bias_y)),
            trainable=True)
        
    def call(self, input1, input2):
        if self.bias_x:
            input1 = tf.concat((input1, tf.ones_like(input1[..., :1])), axis=-1)
        if self.bias_y:
            input2 = tf.concat((input2, tf.ones_like(input2[..., :1])), axis=-1)
        # bxi,oij,byj->boxy
        logits = tf.einsum('bxi,oij,byj->boxy', input1, self.w, input2)
        return logits

class BiaffineAttentionModel(tf.keras.Model):
    def __init__(self, vocab_size, pos_size, embedding_size, num_lstm_units, num_lstm_layers,
                 num_mlt_layers, arc_mlt_size, rel_mlt_size, rel_size, dropout_rate):
        super(BiaffineAttentionModel, self).__init__()

        self.word_embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.pos_embedding = tf.keras.layers.Embedding(pos_size, embedding_size)

        self.lstms = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            num_lstm_units, return_sequences=True)) for _ in range(num_lstm_layers)]
        
        self.arc_head = MLP(arc_mlt_size, num_mlt_layers, dropout_rate)
        self.arc_dep = MLP(arc_mlt_size, num_mlt_layers, dropout_rate)
        self.rel_head = MLP(rel_mlt_size, num_mlt_layers, dropout_rate)
        self.rel_dep = MLP(rel_mlt_size, num_mlt_layers, dropout_rate)
 
        self.arc_biaffine = Biaffine(arc_mlt_size, 1, bias_x=True, bias_y=False)
        self.rel_biaffine = Biaffine(rel_mlt_size, rel_size, bias_x=True, bias_y=True)

        self.embedding_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, word_ids, pos_ids, training=False):
        word_embedding = self.word_embedding(word_ids)
        pos_embedding = self.pos_embedding(pos_ids)
        embedding = tf.concat((word_embedding, pos_embedding), axis=-1)
        self.embedding_dropout(embedding, training=training)
        # bilstm layer
        lstm_output = embedding
        for lstm in self.lstms:
            lstm_output = lstm(lstm_output)
        # mlt and biaffine layer
        # shape=(batch_size, 1, seq_len, seq_len)
        arc_logit = self.arc_biaffine(
            self.arc_dep(lstm_output, training), self.arc_head(lstm_output, training))
        arc_logit = tf.squeeze(arc_logit, axis=1)
        # shape=(batch_size, rel_size, seq_len, seq_len)
        rel_logit = self.rel_biaffine(
            self.rel_dep(lstm_output, training), self.rel_head(lstm_output, training))
        rel_logit = tf.transpose(rel_logit, perm=(0, 2, 3 ,1))
        return arc_logit, rel_logit

class BiaffineAttention(object):
    def __init__(self, vocab_size, pos_size, embedding_size, num_lstm_units, num_lstm_layers,
                 num_mlt_layers, arc_mlt_size, rel_mlt_size, rel_size, learning_rate, adam_beta_2, 
                 dropout_rate):
        self.model = BiaffineAttentionModel(
            vocab_size=vocab_size,
            pos_size=pos_size,
            embedding_size=embedding_size,
            num_lstm_units=num_lstm_units,
            num_lstm_layers=num_lstm_layers,
            num_mlt_layers=num_mlt_layers,
            arc_mlt_size=arc_mlt_size,
            rel_mlt_size=rel_mlt_size,
            rel_size=rel_size,
            dropout_rate=dropout_rate)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(True, reduction='none')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_2=adam_beta_2)
        self.global_step = tf.Variable(0, trainable=False, name='globle_step')

        self.metric_loss = tf.keras.metrics.Mean(name='loss')
        self.metric_uas = tf.keras.metrics.Mean(name='uas')
        self.metric_las = tf.keras.metrics.Mean(name='las')

    @staticmethod
    def get_rel_indices(arc_true):
        batch_size = tf.shape(arc_true)[0]
        seq_len = tf.shape(arc_true)[1]
        index1 = tf.tile(tf.expand_dims(tf.range(batch_size), -1), [1, seq_len])
        index2 = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])
        indices = tf.stack((index1, index2, arc_true), axis=2)
        return indices
    
    def loss_function(self, arc_true, arc_pred, rel_true, rel_pred, mask):
        arc_loss = self.loss_object(arc_true, arc_pred)
        rel_loss = self.loss_object(rel_true, rel_pred)
        # ignore arc and rel of the root word
        loss = tf.boolean_mask((arc_loss + rel_loss)[:, 1:], mask[:, 1:])
        loss = tf.reduce_mean(loss)
        return loss
    

    def get_uas_las(self, arc_true, arc_pred, rel_true, rel_pred, mask):
        # calculate uas
        arc_pred = tf.argmax(arc_pred, -1, output_type=arc_true.dtype)
        arc_pred = tf.boolean_mask(arc_pred[:, 1:], mask[:, 1:])
        arc_true = tf.boolean_mask(arc_true[:, 1:], mask[:, 1:])
        arc_correct = arc_true == arc_pred
        uas = tf.reduce_mean(tf.cast(arc_correct, tf.float32))
        # calculate las
        rel_pred = tf.argmax(rel_pred, -1, output_type=rel_true.dtype)
        rel_pred = tf.boolean_mask(rel_pred[:, 1:], mask[:, 1:])
        rel_true = tf.boolean_mask(rel_true[:, 1:], mask[:, 1:])
        rel_correct = tf.logical_and(rel_true == rel_pred, arc_correct)
        las = tf.reduce_mean(tf.cast(rel_correct, tf.float32))
        return uas, las
    

    def train(self, train_dataset, epochs, batch_size, valid_dataset=None):
        assert isinstance(train_dataset, tf.data.Dataset), 'unknown dataset type!'
        train_dataset = train_dataset.shuffle(20000).batch(batch_size)
        time_record = time.time()

        for epoch in range(epochs):
            for batch_data in train_dataset:
                self.train_step(*batch_data)

                if self.global_step % 10 == 0:
                    print('epoch:%d step:%d sesc:%.2fs loss:%.4f uas:%.4f, las:%.4f' % (
                        epoch, 
                        self.global_step.numpy(), 
                        time.time() - time_record, 
                        self.metric_loss.result().numpy(), 
                        self.metric_uas.result().numpy(),
                        self.metric_las.result().numpy()
                        ))
                    self.metric_loss.reset_states()
                    self.metric_uas.reset_states()
                    self.metric_las.reset_states()
                    time_record = time.time()

            if valid_dataset is not None:
                self.evaluate(valid_dataset)
    


    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32), 
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32), 
        tf.TensorSpec(shape=(None, ), dtype=tf.int32)])
    def train_step(self, input_word, input_pos, input_head, input_rel, input_len):
        mask = tf.sequence_mask(input_len, tf.shape(input_word)[1])
        with tf.GradientTape() as tape:
            arc_logit, rel_logit = self.model(input_word, input_pos, training=True)
            rel_logit = tf.gather_nd(rel_logit, self.get_rel_indices(input_head))
            loss = self.loss_function(input_head, arc_logit, input_rel, rel_logit, mask)

        # apply clipped gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        clipped_grads = [tf.clip_by_value(grad, -2.0, 2.0) for grad in grads]
        self.optimizer.apply_gradients(zip(clipped_grads, self.model.trainable_variables))

        # calculate uas and las
        uas, las = self.get_uas_las(input_head, arc_logit, input_rel, rel_logit, mask)

        # update step
        self.global_step.assign_add(1)

        # record metric
        self.metric_loss(loss)
        self.metric_uas(uas)
        self.metric_las(las)
        return loss, uas, las


    def evaluate(self, dataset, batch_size=64):
        UAS = LAS = 0.
        steps = 0
        for input_word, input_pos, input_head, input_rel, input_len in dataset.batch(batch_size):
            arc_logit, rel_logit = self.model(input_word, input_pos, training=False)
            rel_logit = tf.gather_nd(rel_logit, self.get_rel_indices(input_head))
            # caculate uas and las
            mask = tf.sequence_mask(input_len, tf.shape(input_word)[1])
            uas, las = self.get_uas_las(input_head, arc_logit, input_rel, rel_logit, mask)
            UAS, LAS, steps = UAS + uas.numpy(), LAS + las.numpy(), steps + 1
        print('evaluation uas:%.4f las:%.4f' % (UAS / steps, LAS / steps))





if __name__=='__main__':
    train_data_path='./data/conll/train.conll'
    train_datasets = load(train_data_path)
    word_count = defaultdict(int)
    word_dict = {}
    pos_dict = {}
    rel_dict = {}
    for sentence in train_datasets:
        for token in sentence:
            word_count[token.word] += 1
            pos_dict.setdefault(token.pos, len(pos_dict))
            rel_dict.setdefault(token.rel, len(rel_dict))
    for word, count in word_count.items():
            if count >= 1:
                word_dict.setdefault(word, len(word_dict))
    word_dict.setdefault('<unk>',len(word_dict))
    word_dict.setdefault('<pad>',len(word_dict))
    id2word, _ = zip(*sorted(word_dict.items(), key=lambda x:x[1]))
    id2pos, _ = zip(*sorted(pos_dict.items(), key=lambda x:x[1]))
    id2rel, _ = zip(*sorted(rel_dict.items(), key=lambda x:x[1]))
    inputs = []
    for sentence in train_datasets:
        sentence_input = []
        for token in sentence:
            word_id = word_dict.get(token.word,  word_dict['<unk>'])
            pos_id = pos_dict.get(token.pos, pos_dict['<unk>'])
            head_id =  token.head_id
            rel_id = rel_dict.get(token.rel, rel_dict['<null>'])
            sentence_input.append((word_id, pos_id, head_id, rel_id))
        inputs.append(list(zip(*sentence_input)) + [len(sentence)])
        sentence_input= []
    words, poss, heads, rels, seq_lens = zip(*inputs)
    words = pad_sequences(words, max_len=128,value=word_dict['<pad>'])
    poss = pad_sequences(poss, max_len=128,value=pos_dict['<unk>'])
    heads = pad_sequences(heads, max_len=128,value=0)
    rels = pad_sequences(rels, max_len=128,value=rel_dict['<null>'])
    seq_lens = tf.convert_to_tensor(seq_lens, tf.int32)
    train_dataset = tf.data.Dataset.from_tensor_slices((words, poss, heads, rels, seq_lens))
    val_data_path='./data/conll/dev.conll'
    val_datasets = load(val_data_path)
    inputs = []
    for sentence in val_datasets:
        sentence_input = []
        for token in sentence:
            word_id = word_dict.get(token.word,  word_dict['<unk>'])
            pos_id = pos_dict.get(token.pos, pos_dict['<unk>'])
            head_id =  token.head_id
            rel_id = rel_dict.get(token.rel, rel_dict['<null>'])
            sentence_input.append((word_id, pos_id, head_id, rel_id))
        inputs.append(list(zip(*sentence_input)) + [len(sentence)])
        sentence_input= []
    words, poss, heads, rels, seq_lens = zip(*inputs)
    words = pad_sequences(words, max_len=128,value=word_dict['<pad>'])
    poss = pad_sequences(poss, max_len=128,value=pos_dict['<unk>'])
    heads = pad_sequences(heads, max_len=128,value=0)
    rels = pad_sequences(rels, max_len=128,value=rel_dict['<null>'])
    seq_lens = tf.convert_to_tensor(seq_lens, tf.int32)
    val_dataset = tf.data.Dataset.from_tensor_slices((words, poss, heads, rels, seq_lens))
    print(len(val_dataset))
    print(len(train_dataset))
    for data in train_dataset:
        print(data)
        break
    # model = BiaffineAttention(vocab_size=len(word_dict),pos_size=len(pos_dict),rel_size=len(rel_dict),
    # embedding_size=100,num_lstm_units=400,num_lstm_layers=3,num_mlt_layers=1,arc_mlt_size=500,rel_mlt_size=100,learning_rate=0.002,
    # adam_beta_2=0.9,dropout_rate=0.33)
    # model.train(train_dataset, epochs=15, batch_size=128, valid_dataset=val_dataset)
