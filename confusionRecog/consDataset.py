"""A mini synthetic dataset for graph classification benchmark."""

import os
import sys
from typing import List

import dgl

import numpy as np
import torch as th

from gensim.models import Word2Vec

import tree_sitter
from tree_sitter import Language, Parser

lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
from ParameterConfig import ParameterConfig


class consDataset(object):
    """The dataset class.

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Parameters
    ----------
    unique_fun_file: str
        path of the file storing all the unique functions
    label_maps: dict
        a map consisting of label-str:int-val
    """

    def __init__(self, corpus_path, dic_file_path, label_maps=None):
        super(consDataset, self).__init__()
        self.corpus_path = corpus_path
        self.dic_file_path = dic_file_path
        self.label_maps = label_maps

        self.graphs = []
        self.labels = []
        self.tokens = []
        self.trees = []
        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.trees[idx], self.tokens[idx], self.graphs[idx], self.labels[idx]

    @property
    def num_classes(self):
        """Number of classes."""
        class_set = set(self.labels)
        return len(class_set)

    def _generate(self):
        self.word_index, self.embeddings_matrix = self.construct_ins_embedding()

        trees, tokens, texts, cfgs, labels = self.prepare_data(self.corpus_path, self.word_index, self.label_maps)

        self.tokens = tokens
        self.labels = labels
        self.trees = trees
        self.generate_dglgraph(texts, cfgs)

    def construct_ins_embedding(self):
        """
        construct a vector matrix from pre-trained word2vec models
        :param ins2vec_model:
        :return:
        """
        # 如果没有w2v模型，则训练保存模型
        if not os.path.exists(self.dic_file_path):
            self.get_word2vec_model()

        ins2vec_model = Word2Vec.load(self.dic_file_path)
        vocab_size = len(ins2vec_model.wv.index_to_key)
        print('vocabulary size: {:d}'.format(vocab_size))

        index = 0
        # 存储所有的词语及其索引
        # 初始化 [word : index]
        word_index = {"PAD": index}
        # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于padding补零。
        # 行数为所有单词数+1；比如10000+1；列数为词向量“维度”，比如100。
        if ins2vec_model.vector_size != ParameterConfig.EMBEDDING_DIM:
            print("W2V vector dimension not equal to the configured dimension")
        embeddings_matrix = np.zeros((vocab_size + 2, ins2vec_model.vector_size))

        # 填充上述的字典和矩阵
        for word in ins2vec_model.wv.index_to_key:
            index = index + 1
            word_index[word] = index
            embeddings_matrix[index] = ins2vec_model.wv[word]

        # OOV词随机初始化为同一向量
        index = index + 1
        word_index["UNKNOWN"] = index
        embeddings_matrix[index] = np.random.rand(ins2vec_model.vector_size) / 10

        return word_index, embeddings_matrix

    def prepare_data(self, data_dir, word_index, label_maps):
        """
        Prepare data for compiler family identification task
        Data preparation for other tasks can be easily implemented with slight modifications to this method
        :param data_dir:
        :return:
        """

        def tree_to_index(node):
            if not isinstance(node, tree_sitter.Tree):
                token = node.type
                if len(node.children) == 0:
                    token = node.text
                children = node.children
            else:
                token = node.root_node.type
                if len(node.root_node.children) == 0:
                    token = node.root_node.text
                children = node.root_node.children

            if type(token) is bytes:
                token = token.decode('utf-8')

            result = [word_index[token] if token in word_index else word_index['UNKNOWN']]
            for child in children:
                result.append(tree_to_index(child))
            return result

        fun_rep_file_list = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                fun_rep_file_list.append(file)
        texts = []
        labels = []
        cfgs = []
        tokens = []
        trees = []
        processed = 0
        for fname in fun_rep_file_list:

            processed += 1
            if processed % 100 == 0:
                print('{:.2f} percent files parsed......'.format(processed / len(fun_rep_file_list) * 100))
            # print('Processing '+fname)
            fpath = os.path.join(data_dir, fname)
            # compiler_setting_str = fname.split('#')[2]
            # compiler_options = compiler_setting_str.split('-')
            # compiler_family = compiler_options[0]
            # label_str = compiler_family

            # fname virt1.csv
            label_str = fname[0:-4]
            # print(fname)

            with open(fpath) as f:
                lines = f.readlines()
                indices_set = []  # indices set representation for a function
                token = []
                tree = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('>>>'):
                        if line.startswith('>>>PDG'):
                            edges = line.split(' ')[1:]
                        if line.startswith('>>>Token'):
                            for word in line.split(' ')[1:]:
                                try:
                                    token.append(word_index[word])  # 把句子中的词语转化为index
                                except:
                                    token.append(word_index['UNKNOWN'])  # OOV词统一用'UNKNOWN'对应的向量表示
                            # 读取token转化成树，再进行编码得到树的表述就ok
                            code = line[9:]
                            tree_temp = self.parse_ast(code)
                            tree = tree_to_index(tree_temp)

                        if line.startswith('>>>Func'):
                            # has_cfg = line.split('&')[-1]
                            # if has_cfg != '-1':  # -1 indicates there exists no cfg for the function
                            if len(edges) >= ParameterConfig.CFG_MIN_EDGE_NUM:
                                texts.append(indices_set)
                                # if len(texts) == 1808:
                                #     print(" ".join(edges))
                                #     print(fpath)
                                labels.append(label_maps[label_str])
                                cfgs.append(edges)
                                tokens.append(self.pad_tokens(token))
                                trees.append(tree)
                            indices_set = []
                            token = []
                            tree = []
                        continue
                    # 序号化文本，tokenizer句子，并返回每个句子所对应的词语索引
                    indices = []
                    for word in line.split(' '):
                        try:
                            indices.append(word_index[word])  # 把句子中的词语转化为index
                        except:
                            indices.append(word_index['UNKNOWN'])  # OOV词统一用'UNKNOWN'对应的向量表示
                    indices_set.append(indices)
                f.close()
        return trees, tokens, texts, cfgs, labels

    def generate_dglgraph(self, texts, cfgs):
        '''
        organize all cfgs together with their nodes attributes into dglgraphs
        :param texts: a list consisting of all nodes attributes of the functions
        :param cfgs: a list consisting of all cfgs of all the functions
        :return:
        '''
        assert len(texts) == len(cfgs)
        f_num = len(cfgs)
        # Create the graph from a list of integer pairs.
        elist = []
        for i in range(f_num):
            # create a dgl graph for each cfg
            cfg = cfgs[i]
            for edg in cfg:
                nd_ids = edg.split('->')
                nd_ids = [int(j) for j in nd_ids]
                edg = tuple(nd_ids)
                elist.append(edg)
            dgl_graph = dgl.graph(elist)
            dgl_graph = dgl.add_self_loop(dgl_graph)
            elist = []

            # assign node attributes
            nodes_attrs = texts[i]
            # convert attribute list to torch tensor
            nodes_attrs = self.pad_nodes(nodes_attrs)
            dgl_graph.ndata['w'] = nodes_attrs
            # append to the graph list
            self.graphs.append(dgl_graph)

    def pad_nodes(self, nodes_attrs):
        '''
        pad/truncate all sequences to the same length (specified by ParameterConfig.MAX_SEQUENCE_LENGTH)
        :param nodes_attrs:
        :return: torch tensor
        '''
        torches = []
        for node_atts in nodes_attrs:
            tmp_torch = th.tensor(node_atts)
            actual_len = len(tmp_torch)
            if actual_len > ParameterConfig.MAX_SEQUENCE_LENGTH:
                tmp_torch = tmp_torch[:ParameterConfig.MAX_SEQUENCE_LENGTH]
            else:
                tmp_torch = th.cat([tmp_torch, tmp_torch.new_zeros(ParameterConfig.MAX_SEQUENCE_LENGTH - actual_len)],
                                   0)
            torches.append(tmp_torch)
        return th.stack(torches, 0)

    def pad_tokens(self, token):
        actual_len = len(token)
        if actual_len > ParameterConfig.MAX_TOKEN_LENGTH:
            token = token[:ParameterConfig.MAX_TOKEN_LENGTH]
        else:
            size = ParameterConfig.MAX_TOKEN_LENGTH - actual_len
            zore_list = [0] * size
            token.extend(zore_list)
        return token

    def get_word2vec_model(self):
        words = MyCorpus(self.corpus_path)
        model = Word2Vec(words)
        model.save(self.dic_file_path)

    def parse_ast(self, source):
        C_LANGUAGE = Language('build_languages/my-languages.so', 'c')
        parser = Parser()
        parser.set_language(C_LANGUAGE)  # set the parser for certain language
        # tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
        tree = parser.parse(source.encode('utf-8').decode('ISO-8859-1').encode())
        return tree


class MyCorpus(object):
    """ Data Preparation \n
    gensim’s word2vec expects a sequence of sentences tas its inpu. Each sentence is a list of words (utf8 strings).

    Gensim only requires that the input must provide sentences sequentially, when iterated over. No need to keep everything
    in RAM: we can provide one sentence, process it, forget it, load another sentence...

    Say we want to further preprocess the words from the files — convert to unicode, lowercase, remove numbers, extract
    named entities… All of this can be done inside the MySentences iterator and word2vec doesn’t need to know. All that is
    required is that the input yields one sentence (list of utf8 words) after another.
    """

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def __iter__(self):
        for root, dirs, files in os.walk(self.corpus_path):
            for file in files:
                fpath = os.path.join(root, file)
                with open(fpath) as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line.startswith('>>>Token') or not line.startswith('>>>'):
                            yield line.split(' ')
