import json



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


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task



type2label = {}
with open("../dataset/types.txt", 'r') as f:
    types = f.readlines()
    for line in types:
        t = line.strip()
        type2label[t] = len(type2label)


def read_classification_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['function'].split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=type2label[js['label']]
                )
            )
            if idx + 1 == data_num:
                break
    return examples