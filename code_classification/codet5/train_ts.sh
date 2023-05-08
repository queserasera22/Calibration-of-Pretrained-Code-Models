CUDA_VISIBLE_DEVICES=1 python train_ts.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5_OOD_token/Java250/checkpoint-best-accuracy/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --eval_filename="../dataset/CodeNet_Java250/OOD_token/valid.jsonl" \
    --test_filename="../dataset/CodeNet_Java250/OOD_token/test.jsonl" \
    --num_labels 250 \
    --dataset_name "Java250" \
    --cache_path="./cache/codet5_OOD_token" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --seed 123456
#
CUDA_VISIBLE_DEVICES=1 python train_ts.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5_OOD_CST/Java250/checkpoint-best-accuracy/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --eval_filename="../dataset/CodeNet_Java250/OOD_CST/valid.jsonl" \
    --test_filename="../dataset/CodeNet_Java250/OOD_CST/test.jsonl" \
    --num_labels 250 \
    --dataset_name "Java250" \
    --cache_path="./cache/codet5_OOD_CST" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --seed 123456
#
CUDA_VISIBLE_DEVICES=2 python train_ts.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5_OOD/Java250/checkpoint-best-accuracy/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --eval_filename="../dataset/CodeNet_Java250/OOD/valid.jsonl" \
    --test_filename="../dataset/CodeNet_Java250/OOD/test.jsonl" \
    --num_labels 250 \
    --dataset_name "Java250" \
    --cache_path="./cache/codet5_OOD" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --seed 123456
#
#CUDA_VISIBLE_DEVICES=0 python train_ts.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --load_model_path="models/codet5_OOD_token/Python800/checkpoint-best-accuracy/pytorch_model.bin" \
#    --model_type codet5 \
#    --task classification \
#    --eval_filename="../dataset/CodeNet_Python800/OOD_token/valid.jsonl" \
#    --test_filename="../dataset/CodeNet_Python800/OOD_token/test.jsonl" \
#    --num_labels 800 \
#    --dataset_name "Python800" \
#    --cache_path="./cache/codet5_OOD_token" \
#    --eval_batch_size=16 \
#    --max_source_length=512 \
#    --seed 123456

CUDA_VISIBLE_DEVICES=0 python train_ts.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5_OOD/Python800/checkpoint-best-accuracy/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --eval_filename="../dataset/CodeNet_Python800/OOD/valid.jsonl" \
    --test_filename="../dataset/CodeNet_Python800/OOD/test.jsonl" \
    --num_labels 800 \
    --dataset_name "Python800" \
    --cache_path="./cache/codet5_OOD" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --seed 123456
