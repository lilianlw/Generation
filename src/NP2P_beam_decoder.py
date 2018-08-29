# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np

from vocab_utils import Vocab
import namespace_utils
import NP2P_data_stream
from NP2P_model_graph import ModelGraph

import re

import tensorflow as tf
import NP2P_trainer

import json
import math

tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL

def search(sess, model, vocab, batch, options, decode_mode='greedy'):
    '''
    for greedy search, multinomial search
    '''
    # Run the encoder to get the encoder hidden states and decoder initial state
    (phrase_representations, initial_state, encoder_features,phrase_idx, phrase_mask) = model.run_encoder(sess, batch, options)
    # phrase_representations: [batch_size, passage_len, encode_dim]
    # initial_state: a tupel of [batch_size, gen_dim]
    # encoder_features: [batch_size, passage_len, attention_vec_size]
    # phrase_idx: [batch_size, passage_len]
    # phrase_mask: [batch_size, passage_len]

    word_t = batch.gen_input_words[:,0]
    state_t = initial_state
    context_t = np.zeros([batch.batch_size, model.encode_dim])
    coverage_t = np.zeros((batch.batch_size, phrase_representations.shape[1]))
    generator_output_idx = [] # store phrase index prediction
    text_results = []
    generator_input_idx = [word_t] # store word index
    for i in xrange(options.max_answer_len):
        if decode_mode == "pointwise": word_t = batch.gen_input_words[:,i]
        feed_dict = {}
        feed_dict[model.init_decoder_state] = state_t
        feed_dict[model.context_t_1] = context_t
        feed_dict[model.coverage_t_1] = coverage_t
        feed_dict[model.word_t] = word_t

        feed_dict[model.phrase_representations] = phrase_representations
        feed_dict[model.encoder_features] = encoder_features
        feed_dict[model.phrase_idx] = phrase_idx
        feed_dict[model.phrase_mask] = phrase_mask
        if options.with_phrase_projection:
            feed_dict[model.max_phrase_size] = batch.max_phrase_size
            if options.add_first_word_prob_for_phrase:
                feed_dict[model.in_passage_words] = batch.sent1_word
                feed_dict[model.phrase_starts] = batch.phrase_starts



        if decode_mode in ["greedy","pointwise"]:
            prediction = model.greedy_prediction
        elif decode_mode == "multinomial":
            prediction = model.multinomial_prediction

        (state_t, context_t, attn_dist_t, coverage_t, prediction) = sess.run([model.state_t, model.context_t, model.attn_dist_t,
                                                                 model.coverage_t, prediction], feed_dict)
        attn_idx = np.argmax(attn_dist_t, axis=1) # [batch_size]
        # convert prediction to word ids
        generator_output_idx.append(prediction)
        prediction = np.reshape(prediction, [prediction.size, 1])
        [cur_words, cur_word_idx] = batch.map_phrase_idx_to_text(prediction) # [batch_size, 1]
        cur_word_idx = np.array(cur_word_idx)
        cur_word_idx = np.reshape(cur_word_idx, [cur_word_idx.size])
        word_t = cur_word_idx
        cur_words = flatten_words(cur_words)

        for i, wword in enumerate(cur_words):
            if wword == 'UNK' and attn_idx[i] < len(batch.passage_words[i]):
                cur_words[i] = batch.passage_words[i][attn_idx[i]]

        text_results.append(cur_words)
        generator_input_idx.append(cur_word_idx)

    generator_input_idx = generator_input_idx[:-1] # remove the last word to shift one position to the right
    generator_output_idx = np.stack(generator_output_idx, axis=1) # [batch_size, max_len]
    generator_input_idx = np.stack(generator_input_idx, axis=1) # [batch_size, max_len]

    prediction_lengths = [] # [batch_size]
    sentences = [] # [batch_size]
    for i in xrange(batch.batch_size):
        words = []
        for j in xrange(options.max_answer_len):
            cur_phrase = text_results[j][i]
#             cur_phrase = cur_batch_text[j]
            words.append(cur_phrase)
            if cur_phrase == "</s>": break# filter out based on end symbol
        prediction_lengths.append(len(words))
        cur_sent = " ".join(words)
        sentences.append(cur_sent)

    return (sentences, prediction_lengths, generator_input_idx, generator_output_idx)

def flatten_words(cur_words):
    all_words = []
    for i in xrange(len(cur_words)):
        all_words.append(cur_words[i][0])
    return all_words

class Hypothesis(object):
    def __init__(self, tokens, log_ps, state, context_vector, coverage_vector=None):
        self.tokens = tokens # store all tokens
        self.log_probs = log_ps # store log_probs for each time-step

        self.state = state
        self.context_vector = context_vector
        self.coverage_vector = coverage_vector

    def extend(self, token, log_prob, state, context_vector, coverage_vector=None):
        return Hypothesis(self.tokens + [token], self.log_probs + [log_prob], state,
                          context_vector, coverage_vector=coverage_vector)

    def latest_token(self):
        return self.tokens[-1]

    def avg_log_prob(self):
        return np.sum(self.log_probs[1:])/ (len(self.tokens)-1)

    def probs2string(self):
        out_string = ""
        for prob in self.log_probs:
            out_string += " %.4f" % prob
        return out_string.strip()

    def idx_seq_to_string(self, passage, id2phrase, vocab, options):
        word_size = vocab.vocab_size + 1
        all_words = []
        for idx in self.tokens:
            if idx<word_size:
                cur_word = vocab.getWord(idx)
            else:
                cur_word = id2phrase[idx]
                if not options.withTextChunk:
                    items = re.split('-', cur_word)
                    cur_word = passage.getTokChunk(int(items[0]), int(items[1]))
            all_words.append(cur_word)
        return " ".join(all_words[1:])


def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)



def run_beam_search(sess, model, vocab, batch, options):
    # Run encoder
    (phrase_representations, initial_state, encoder_features,phrase_idx, phrase_mask) = model.run_encoder(sess, batch, options)
    # phrase_representations: [1, passage_len, encode_dim]
    # initial_state: a tupel of [1, gen_dim]
    # encoder_features: [1, passage_len, attention_vec_size]
    # phrase_idx: [1, passage_len]
    # phrase_mask: [1, passage_len]

    sent_stop_id = vocab.getIndex('</s>')

    # Initialize this first hypothesis
    context_t = np.zeros([model.encode_dim]) # [encode_dim]
    coverage_t = np.zeros((phrase_representations.shape[1])) # [passage_len]
    hyps = []
    hyps.append(Hypothesis([batch.gen_input_words[0][0]], [0.0], initial_state, context_t, coverage_vector=coverage_t))

    # beam search decoding
    results = [] # this will contain finished hypotheses (those that have emitted the </s> token)
    steps = 0
    while steps < options.max_answer_len and len(results) < options.beam_size:
        cur_size = len(hyps) # current number of hypothesis in the beam
        cur_phrase_representations = np.tile(phrase_representations, (cur_size, 1, 1))
        cur_encoder_features = np.tile(encoder_features, (cur_size, 1, 1)) # [batch_size,passage_len, options.attention_vec_size]
        cur_phrase_idx = np.tile(phrase_idx, (cur_size, 1)) # [batch_size, passage_len]
        cur_phrase_mask = np.tile(phrase_mask, (cur_size, 1)) # [batch_size, passage_len]
        ###uuuzi
        cur_template_words = np.tile(batch.template_word,(cur_size,1))
        cur_template_lengths = np.tile(batch.template_length,(cur_size))

        cur_state_t_1 = [] # [2, gen_steps]
        cur_context_t_1 = [] # [batch_size, encoder_dim]
        cur_coverage_t_1 = [] # [batch_size, passage_len]
        cur_word_t = [] # [batch_size]
        for h in hyps:
            cur_state_t_1.append(h.state)
            cur_context_t_1.append(h.context_vector)
            cur_word_t.append(h.latest_token())
            cur_coverage_t_1.append(h.coverage_vector)
        cur_context_t_1 = np.stack(cur_context_t_1, axis=0)
        cur_coverage_t_1 = np.stack(cur_coverage_t_1, axis=0)
        cur_word_t = np.array(cur_word_t)

        cells = [state.c for state in cur_state_t_1]
        hidds = [state.h for state in cur_state_t_1]
        new_c = np.concatenate(cells, axis=0)
        new_h = np.concatenate(hidds, axis=0)
        new_dec_init_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h) ###

        feed_dict = {}
        feed_dict[model.init_decoder_state] = new_dec_init_state
        feed_dict[model.context_t_1] = cur_context_t_1
        feed_dict[model.word_t] = cur_word_t

        feed_dict[model.phrase_representations] = cur_phrase_representations
        feed_dict[model.encoder_features] = cur_encoder_features
        feed_dict[model.phrase_idx] = cur_phrase_idx
        feed_dict[model.phrase_mask] = cur_phrase_mask
        feed_dict[model.coverage_t_1] = cur_coverage_t_1
        if options.with_phrase_projection:
            feed_dict[model.max_phrase_size] = batch.max_phrase_size
            if options.add_first_word_prob_for_phrase:
                feed_dict[model.in_passage_words] = batch.sent1_word
                feed_dict[model.phrase_starts] = batch.phrase_starts
        if options.with_template: ###
            feed_dict[model.template_lengths] = cur_template_lengths
            feed_dict[model.template_words] = cur_template_words

        (state_t, context_t, coverage_t, topk_log_probs, topk_ids) = sess.run([model.state_t, model.context_t,
                                                                 model.coverage_t, model.topk_log_probs, model.topk_ids], feed_dict)

        new_states = [tf.contrib.rnn.LSTMStateTuple(state_t.c[i:i+1, :], state_t.h[i:i+1, :]) for i in xrange(cur_size)] ###

        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        for i in xrange(cur_size):
            h = hyps[i]
            cur_state = new_states[i]
            cur_context = context_t[i]
            cur_coverage = coverage_t[i]
            for j in xrange(options.beam_size):
                cur_tok = topk_ids[i, j]
                cur_tok_log_prob = topk_log_probs[i, j]
                new_hyp = h.extend(cur_tok, cur_tok_log_prob, cur_state, cur_context, coverage_vector=cur_coverage)
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        # hyps will contain hypotheses for the next step
        hyps = []
        for h in sort_hyps(all_hyps):
            # If this hypothesis is sufficiently long, put in results. Otherwise discard.
            if h.latest_token() == sent_stop_id:
                if steps >= options.min_answer_len:
                    results.append(h)
            # hasn't reached stop token, so continue to extend this hypothesis
            else:
                hyps.append(h)
            if len(hyps) == options.beam_size or len(results) == options.beam_size:
                break

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps
    # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    if len(results)==0:
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)

    # Return the hypothesis with highest average log prob
    return hyps_sorted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='The path to the test file.')
    parser.add_argument('--out_path', type=str, help='The path to the output file.')
    parser.add_argument('--mode', type=str,default='pointwise', help='The path to the output file.')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_prefix
    in_path = args.in_path
    out_path = args.out_path
    mode = args.mode

    ### print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    # load the configuration file
    print('Loading configurations from ' + model_prefix + ".config.json")
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
    FLAGS = NP2P_trainer.enrich_options(FLAGS)

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    # load vocabs
    print('Loading vocabs.')
    word_vocab = char_vocab = POS_vocab = NER_vocab = None
    if FLAGS.with_word:
        word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
        print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    if FLAGS.with_char:
        char_vocab = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    if FLAGS.with_POS:
        POS_vocab = Vocab(model_prefix + ".POS_vocab", fileformat='txt2')
        print('POS_vocab: {}'.format(POS_vocab.word_vecs.shape))
    if FLAGS.with_NER:
        NER_vocab = Vocab(model_prefix + ".NER_vocab", fileformat='txt2')
        print('NER_vocab: {}'.format(NER_vocab.word_vecs.shape))
    if FLAGS.with_template:
        template_vocab = Vocab(model_prefix + ".template_vocab", fileformat='txt2')
        print('template_vocab: {}'.format(template_vocab.word_vecs.shape))

    # template calculation preparation
    interrogative_words = ["which","that","what","who","whose","how","where","times","many","long","much","often","year"]
    def question_dict_pattern(annotation):
        tokens = annotation["toks"].strip().split(" ")
        pos_tags = annotation["POSs"].strip().split(" ")
        negs = annotation['NERs'].strip().split(" ")
        if len(tokens)!=len(pos_tags):
            print("mismatch length!")
        template = []
        for i in range(len(tokens)):
            if tokens[i].lower() in interrogative_words:
                template.append(tokens[i])
            elif negs[i] != "O":
                template.append(negs[i])
            else:
                template.append(pos_tags[i])
        return template

    word_vecs_dict = {}
    word_dim = 0
    with open( FLAGS.word_vec_path,"r") as vf:
        lines = vf.readlines()
        for line in lines:
            word = line.strip().split('\t')[1]
            vecs = line.strip().split('\t')[2].split(' ')
            word_vecs_dict[word] = vecs
        word_dim = len(vecs)

    paragraphs = {}
    answers = {}
    questions = {}
    questions_templates = {}
    answer_vecs_dict = {}
    paragraph_vecs_dict = {}

    with open(FLAGS.train_path,"r") as f:
        data = json.load(f) #list len(data):75722
        #data[0] #dict_keys(['text3', 'text1', 'text2', 'annotation3', 'id', 'annotation2', 'annotation1'])  id,paragraph,question,answer
        id_num = 0
        for item in data:
            _id = id_num + 1 #item["id"]
            paragraphs[_id] = item["annotation1"] #item["text1"]
            questions[_id] = item["annotation2"]
            questions_templates[_id] = question_dict_pattern(item["annotation2"])    #question_template[_id] = question_pattern(item["text2"])              
            answers[_id] = item["annotation3"]
            ## answer vecs
            m = 0
            vecs = np.zeros(( word_dim ),np.float)
            for tok in item["annotation3"]["toks"].strip().split(" "):
                if tok in word_vecs_dict:
                    vecs = vecs + np.array(word_vecs_dict[tok]).astype(np.float)
                    m = m + 1
                    if m!=0:
                        vecs = vecs / m
            answer_vecs_dict[_id] = vecs.tolist()
            ### paragraph vecs
            m = 0
            vecs = np.zeros(( word_dim ),np.float)
            for tok in item["annotation1"]["toks"].strip().split(" "):
                if tok in word_vecs_dict:
                    vecs = vecs + np.array(word_vecs_dict[tok]).astype(np.float)
                    m = m + 1
                    if m!=0:
                        vecs = vecs / m
            paragraph_vecs_dict[_id] = vecs.tolist()
    def answer_dict_pattern(annotation):
        tokens = annotation["toks"].strip().split(" ")
        pos_tags = annotation["POSs"].strip().split(" ")
        negs = annotation['NERs'].strip().split(" ")
        if len(tokens)!=len(pos_tags):
            print("mismatch length!")
        template = []
        for i in range(len(tokens)):
            if ( pos_tags[i]=="CD" and re.match(r'\d{4}', tokens[i]) is not None and len(tokens[i])==4 ): #re.match(r'\d{4}', tokens[i]).span()
                template.append("YEAR")
            elif negs[i] != "O":
                template.append(negs[i])
            else:
                template.append(pos_tags[i])
        return template
    def cosine_similarity(v1,v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)
    def bow_similarity(annotation, answer_annotation):
        #pos = annotation["POSs"].split(" ")
        template_candidate = answer_dict_pattern(annotation)
        template_answer = answer_dict_pattern(answer_annotation)
        vocab = {}
        cnt = 0
        for item in (template_candidate+template_answer):
            if item not in vocab:
                vocab[item] = cnt
                cnt = cnt + 1
        answer_vec = []
        vec = []
        for i in range(len(vocab)):
            answer_vec.append(0)
            vec.append(0)
        for item in template_candidate:
            answer_vec[vocab[item]] += 1
        for item in template_answer:
            vec[vocab[item]] += 1
        score = cosine_similarity(answer_vec, vec)    
        return score
    def cos_sim(a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according 
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        consine_similarity = 0
        if norm_a!=0 and norm_b!=0 :
            consine_similarity = dot_product / (norm_a * norm_b)
        return consine_similarity
    def semantic_similarity_answer(_id, annotation):
        tokens = annotation["toks"].strip().split(" ")
        vecs = np.zeros((dim),np.float)
        m = 0
        for tok in tokens:
            if tok in word_vecs_dict:
                vecs = vecs + np.array(word_vecs_dict[tok]).astype(np.float)
                m = m + 1
        if m!=0:
            vecs = vecs / m
        vecs_candidate = np.array(answer_vecs_dict.get(_id)).astype(np.float)
        score = cos_sim(vecs, vecs_candidate)
        return score

    def semantic_similarity_paragraph(_id, annotation):
        tokens = annotation["toks"].strip().split(" ")
        vecs = np.zeros((dim),np.float)
        m = 0
        for tok in tokens:
            if tok in word_vecs_dict:
                vecs = vecs + np.array(word_vecs_dict[tok]).astype(np.float)
                m = m + 1
        if m!=0:
            vecs = vecs / m
        vecs_candidate = np.array(paragraph_vecs_dict.get(_id)).astype(np.float)
        score = cos_sim(vecs, vecs_candidate)
        return score 

    print('Loading test set.')
    if FLAGS.infile_format == 'fof':
        testset, _ = NP2P_data_stream.read_generation_datasets_from_fof(in_path, isLower=FLAGS.isLower)
    elif FLAGS.infile_format == 'plain':
        testset, _ = NP2P_data_stream.read_all_GenerationDatasets(in_path, isLower=FLAGS.isLower)
    else:
        testset, _ = NP2P_data_stream.read_all_GQA_questions(in_path, isLower=FLAGS.isLower, switch=FLAGS.switch_qa)
    print('Number of samples: {}'.format(len(testset)))

    if FLAGS.with_template:
        if FLAGS.template_train_retrieval:
            for (paragraph, question, answer) in testset:
                answer_annotation = answer.annotation
                paragraph_annotation = paragraph.annotation
                template_answer = answer_dict_pattern(answer_annotation)
                score_dict = {}
                for _id in answers.keys():
                    score_dict[_id] = bow_similarity(answers[_id],answer_annotation) + semantic_similarity_answer(_id,answer_annotation) + semantic_similarity_paragraph(_id,paragraph_annotation)
                score_list = sorted(score_dict.items(), key = lambda x: x[1], reverse=True)
                question.template = ' '.join(questions_templates[_id])
                question.template_length = len(questions_templates[_id])
        else:
            for (paragraph, question, answer) in testset:
                question.template = ' '.join(question_dict_pattern(question.annotation))
                question.template_length = len(question_dict_pattern(question.annotation))

    print('Build DataStream ... ')
    batch_size=-1
    if mode in ['beam_search', 'beam_evaluate']: batch_size = 1
    devDataStream = NP2P_data_stream.QADataStream(testset, word_vocab, char_vocab, POS_vocab, NER_vocab, template_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=batch_size)
    print('Number of instances in testDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(devDataStream.get_num_batch()))

    best_path = model_prefix + ".best.model"
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                valid_graph = ModelGraph(template_vocab=template_vocab, word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab,
                                         NER_vocab=NER_vocab, options=FLAGS, mode="decode")

        ## remove word _embedding
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        initializer = tf.global_variables_initializer()
        ### sess = tf.Session()
        ### sess.run(initializer)
        config_gpu = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) ###
        config_gpu.gpu_options.allow_growth = True ###
        sess = tf.Session(config=config_gpu) ###
        sess.run(initializer)

        saver.restore(sess, best_path) # restore the model

        total = 0
        correct = 0
        if mode.endswith('evaluate'):
            ref_outfile = open(out_path+ ".ref", 'wt')
            pred_outfile = open(out_path+ ".pred", 'wt')
        else:
            outfile = open(out_path, 'wt')
        total_num = devDataStream.get_num_batch()
        devDataStream.reset()
        for i in range(total_num):
            cur_batch = devDataStream.get_batch(i)
            if mode == 'pointwise':
                (sentences, prediction_lengths, generator_input_idx,
                 generator_output_idx) = search(sess, valid_graph, word_vocab, cur_batch, FLAGS, decode_mode=mode)
                for j in xrange(cur_batch.batch_size):
                    cur_total = cur_batch.answer_lengths[j]
                    cur_correct = 0
                    for k in xrange(cur_total):
                        if generator_output_idx[j,k]== cur_batch.in_answer_words[j,k]: cur_correct+=1.0
                    total += cur_total
                    correct += cur_correct
                    outfile.write(cur_batch.instances[j][1].tokText.encode('utf-8') + "\n")
                    outfile.write(sentences[j].encode('utf-8') + "\n")
                    outfile.write("========\n")
                outfile.flush()
                print('Current dev accuracy is %d/%d=%.2f' % (correct, total, correct/ float(total) * 100))
            elif mode in ['greedy', 'multinomial']:
                print('Batch {}'.format(i))
                (sentences, prediction_lengths, generator_input_idx,
                 generator_output_idx) = search(sess, valid_graph, word_vocab, cur_batch, FLAGS, decode_mode=mode)
                for j in xrange(cur_batch.batch_size):
                    outfile.write(cur_batch.instances[j][1].ID_num.encode('utf-8') + "\n")
                    outfile.write(cur_batch.instances[j][1].tokText.encode('utf-8') + "\n")
                    outfile.write(sentences[j].encode('utf-8') + "\n")
                    outfile.write("========\n")
                outfile.flush()
            elif mode == 'greedy_evaluate':
                print('Batch {}'.format(i))
                (sentences, prediction_lengths, generator_input_idx,
                generator_output_idx) = search(sess, valid_graph, word_vocab, cur_batch, FLAGS, decode_mode="greedy")
                for j in xrange(cur_batch.batch_size):
                    ref_outfile.write(cur_batch.instances[j][1].tokText.encode('utf-8') + "\n")
                    pred_outfile.write(sentences[j].encode('utf-8') + "\n")
                ref_outfile.flush()
                pred_outfile.flush()
            elif mode == 'beam_evaluate':
                print('Instance {}'.format(i))
                ref_outfile.write(cur_batch.instances[0][1].tokText.encode('utf-8') + "\n")
                ref_outfile.flush()
                hyps = run_beam_search(sess, valid_graph, word_vocab, cur_batch, FLAGS)
                cur_passage = cur_batch.instances[0][0]
                cur_id2phrase = None
                if FLAGS.with_phrase_projection: (cur_phrase2id, cur_id2phrase) = cur_batch.phrase_vocabs[0]
                cur_sent = hyps[0].idx_seq_to_string(cur_passage, cur_id2phrase, word_vocab, FLAGS)
                pred_outfile.write(cur_sent.encode('utf-8') + "\n")
                pred_outfile.flush()
            else: # beam search
                print('Instance {}'.format(i))
                hyps = run_beam_search(sess, valid_graph, word_vocab, cur_batch, FLAGS)
                outfile.write("Input: " + cur_batch.instances[0][0].tokText.encode('utf-8') + "\n")
                outfile.write("Truth: " + cur_batch.instances[0][1].tokText.encode('utf-8') + "\n")
                for j in xrange(len(hyps)):
                    hyp = hyps[j]
                    cur_passage = cur_batch.instances[0][0]
                    cur_id2phrase = None
                    if FLAGS.with_phrase_projection: (cur_phrase2id, cur_id2phrase) = cur_batch.phrase_vocabs[0]
                    cur_sent = hyp.idx_seq_to_string(cur_passage, cur_id2phrase, word_vocab, FLAGS)
                    outfile.write("Hyp-{}: ".format(j) + cur_sent.encode('utf-8') + " {}".format(hyp.avg_log_prob()) + "\n")
                outfile.write("========\n")
                outfile.flush()
        if mode.endswith('evaluate'):
            ref_outfile.close()
            pred_outfile.close()
        else:
            outfile.close()




