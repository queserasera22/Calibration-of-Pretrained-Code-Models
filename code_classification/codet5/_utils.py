import json


# def add_lang_by_task(target_str, task, sub_task):
#     if task == 'summarize':
#         target_str = '<en> ' + target_str
#     elif task == 'refine':
#         target_str = '<java> ' + target_str
#     elif task == 'translate':
#         if sub_task == 'java-cs':
#             target_str = '<c_sharp> ' + target_str
#         else:
#             target_str = '<java> ' + target_str
#     elif task == 'concode':
#         target_str = '<java> ' + target_str
#     elif task == 'defect':
#         target_str = target_str
#     return target_str


# def convert_examples_to_features(item):
#     example, example_index, tokenizer, args, stage = item
#
#     if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
#         if args.sub_task != 'none':
#             source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
#         else:
#             source_str = "{}: {}".format(args.task, example.source)
#     else:
#         source_str = example.source
#
#     source_str = source_str.replace('</s>', '<unk>')
#     source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
#     assert source_ids.count(tokenizer.eos_token_id) == 1
#     if stage == 'test':
#         target_ids = []
#     else:
#         target_str = example.target
#         if args.add_lang_ids:
#             target_str = add_lang_by_task(example.target, args.task, args.sub_task)
#         if args.task in ['defect', 'clone']:
#             if target_str == 0:
#                 target_str = 'false'
#             elif target_str == 1:
#                 target_str = 'true'
#             else:
#                 raise NameError
#         target_str = target_str.replace('</s>', '<unk>')
#         target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
#                                       truncation=True)
#         assert target_ids.count(tokenizer.eos_token_id) == 1
#
#     return InputFeatures(
#         example_index,
#         source_ids,
#         target_ids,
#         url=example.url
#     )




def convert_classification_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return ClassificationInputFeatures(example_index, code, example.target)



class ClassificationInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


# class InputFeatures(object):
#     """A single training/test features for a example."""
#
#     def __init__(self,
#                  example_id,
#                  source_ids,
#                  target_ids,
#                  url=None
#                  ):
#         self.example_id = example_id
#         self.source_ids = source_ids
#         self.target_ids = target_ids
#         self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 # idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        # self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


def read_classification_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['code'].split())
            examples.append(
                Example(
                    # idx=js['index'],
                    source=code,
                    target=js['label']
                )
            )
            if idx + 1 == data_num:
                break
    return examples