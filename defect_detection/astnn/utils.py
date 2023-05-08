import os
import re
from io import StringIO
import tokenize
import random

import numpy as np
import pandas as pd
import torch
from tree_sitter import Node, Language, Parser
from torch.utils.data import Dataset

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class BlockNode(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = get_token(node)
        self.children = self.add_children()

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children) == 0

    def ori_children(self, root):
        if isinstance(root, Node):
            if self.token in ['method_declaration', "class_declaration", 'for_statement', 'while_statement', 'do_statement','switch_statement', 'if_statement']:
                children = root.children[:-1]
            else:
                children = root.children
        else:
            children = []


        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    print("ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def add_children(self):
        if self.is_str:
            return []
        children = self.ori_children(self.node)
        return [BlockNode(child) for child in children]


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item["func"])
        labels.append(item["target"])
    return data, torch.LongTensor(labels)


def get_token(node):
    if isinstance(node, str):
        token = node
    elif isinstance(node, Node):
        if len(node.children) == 0:
            token = str(node.text, 'utf-8')
        else:
            token = node.type
    else:
        token = ''
    return token


def get_sequence(node, sequence):
    token = get_token(node)
    children = node.children
    sequence.append(token)

    for child in children:
        get_sequence(child, sequence)

    if token in ['for_statement', 'while_statement', 'do_statement','switch_statement', 'if_statement']:
        sequence.append('End')


def get_blocks(node, block_seq):
    name = get_token(node)
    children = node.children
    # if len(children) != 0:
    #     print(name, "-------------")
    #     print(children)

    logic = ['for_statement', 'while_statement', 'do_statement', 'switch_statement', 'if_statement']

    if name in ['import_declaration', 'local_variable_declaration', 'method_declaration', 'class_declaration', 'expression_statement']:
        block_seq.append(BlockNode(node))
        for child in children:
            get_blocks(child, block_seq)
    elif name in logic:
        block_seq.append(BlockNode(node))
        for child in children:
            token = get_token(child)
            if token not in ['local_variable_declaration']:
                get_blocks(child, block_seq)
        block_seq.append(BlockNode('End'))
    elif name == 'block':
        block_seq.append(BlockNode(name))
        for child in children:
            get_blocks(child, block_seq)
    else:
        for child in children:
            get_blocks(child, block_seq)


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def parse_data(data_file, lang, max_len=512):
    print("Parsing data...", data_file)

    source = pd.read_json(data_file, lines=True)
    print(len(source['func']))

    def parse_program(func):
        func = func.replace("\\n", "\n")
        try:
            func = remove_comments_and_docstrings(func, lang)
        except:
            pass

        # 防止内存爆炸举措1：限制最多max_len个token
        func = func.strip().split()
        if len(func) > max_len:
            func = func[:max_len]
        func = " ".join(func)
        parser = Parser()
        parser.set_language(Language("../../parser/parser_folder/my-languages.so", lang))
        tree = parser.parse(bytes(func, 'utf8'))
        root = tree.root_node
        return root

    source['func'] = source['func'].apply(parse_program)
    return source

# js2 = {"index": "3142", "code": "import java.util.*;\nimport java.io.InputStreamReader;\n"
#                                 "import java.io.BufferedReader;\n\n\npublic class Main {\n    "
#                                 "public static void main(String[] args) throws Exception {\n"
#                                 "        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));\n"
#                                 "        String s = br.readLine();\n"
#                                 "        \n"
#                                 "        int count1 = 0;\n"
#                                 "        for (int i = 0; i < s.length(); i++) {\n"
#                                 "            if (i % 2 == 0) {\n"
#                                 "                if (s.charAt(i) != '0') count1++;\n"
#                                 "            }\n"
#                                 "            else {\n"
#                                 "                if (s.charAt(i) != '1') count1++;\n"
#                                 "            }\n"
#                                 "        }\n"
#                                 "        \n"
#                                 "        int count2 = 0;\n"
#                                 "        for (int i = 0; i < s.length(); i++) {\n"
#                                 "            if (i % 2 == 0) {\n"
#                                 "                if (s.charAt(i) != '1') count2++;\n"
#                                 "            }\n"
#                                 "            else {\n"
#                                 "                if (s.charAt(i) != '0') count2++;\n"
#                                 "            }\n"
#                                 "        }\n"
#                                 "        \n"
#                                 "        //System.out.println(count2);\n"
#                                 "        System.out.println(Math.min(count1, count2));\n"
#                                 "    }\n}\n\n\n\n", "label": 10}
