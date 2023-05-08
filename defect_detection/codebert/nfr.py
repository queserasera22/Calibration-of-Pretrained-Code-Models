import pandas as pd


# pd 读取 csv 文件
codebert_pred = pd.read_csv('models/codebert/checkpoint-best-acc/eval_pred.csv')
codeberta_pred = pd.read_csv('models/codeberta/checkpoint-best-acc/eval_pred.csv')


codebert_results = (codebert_pred['label'] == codebert_pred['pred']).astype(int)
print("codebert acc: ", codebert_results.mean())


codeberta_results = (codeberta_pred['label'] == codeberta_pred['pred']).astype(int)
print("codeberta acc: ", codeberta_results.mean())

#  codeberta --> codebert
flip_results = codebert_results - codeberta_results

total = len(flip_results)
print("Total: ", total)

count = flip_results.value_counts()

print(count)

print("count_positive: ", count[1] / total)
print("count_negative: ", count[-1] / total)