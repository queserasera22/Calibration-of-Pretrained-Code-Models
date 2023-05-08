CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/POJ104/checkpoint-best-accuracy/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --test_filename="../dataset/POJ104/test.jsonl" \
    --num_labels 104 \
    --dataset_name='POJ104' \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="POJ104-CodeT5"\
    --seed 123456

CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/Java250/checkpoint-best-accuracy/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --test_filename="../dataset/CodeNet_Java250/test.jsonl" \
    --num_labels 250 \
    --dataset_name='Java250' \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Java250-CodeT5"\
    --seed 123456

CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5_OOD/Java250/checkpoint-best-accuracy/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --test_filename="../dataset/CodeNet_Java250/OOD/test.jsonl" \
    --num_labels 250 \
    --dataset_name='Java250' \
    --cache_path="./cache/codet5_OOD" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Java250-CodeT5-OOD"\
    --seed 123456

CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/Python800/checkpoint-best-accuracy/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --test_filename="../dataset/CodeNet_Python800/test.jsonl" \
    --num_labels 800 \
    --dataset_name='Python800' \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Python800-CodeT5"\
    --seed 123456

#CUDA_VISIBLE_DEVICES=3 python ece.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --load_model_path="models/codet5_OOD_CST/Python800/checkpoint-best-accuracy/pytorch_model.bin" \
#    --model_type codet5 \
#    --task classification \
#    --test_filename="../dataset/CodeNet_Python800/OOD_CST/test.jsonl" \
#    --num_labels 800 \
#    --dataset_name='Python800' \
#    --cache_path="./cache/codet5_OOD_CST" \
#    --eval_batch_size=16 \
#    --max_source_length=512 \
#    --save_name="Python800-CodeT5-OOD-CST"\
#    --seed 123456


#CUDA_VISIBLE_DEVICES=0 python ece.py \
#    --model_name_or_path='Salesforce/codet5-base' \
#    --config_name='Salesforce/codet5-base' \
#    --tokenizer_name='Salesforce/codet5-base' \
#    --load_model_path="models/codet5_OOD/Python800/checkpoint-best-accuracy/pytorch_model.bin" \
#    --model_type codet5 \
#    --task classification \
#    --test_filename="../dataset/CodeNet_Python800/OOD/test.jsonl" \
#    --num_labels 800 \
#    --dataset_name='Python800' \
#    --cache_path="./cache/codet5_OOD" \
#    --eval_batch_size=16 \
#    --max_source_length=512 \
#    --save_name="Python800-CodeT5-OOD"\
#    --seed 123456


#CUDA_VISIBLE_DEVICES=2 python ece.py \
#    --model_name_or_path='Salesforce/codet5-large' \
#    --config_name='Salesforce/codet5-large' \
#    --tokenizer_name='Salesforce/codet5-large' \
#    --load_model_path="models/codet5_large/Python800_test/checkpoint-best-accuracy/pytorch_model.bin" \
#    --model_type codet5 \
#    --task classification \
#    --test_filename="../dataset/CodeNet_Python800/test.jsonl" \
#    --num_labels 104 \
#    --dataset_name='Python800' \
#    --cache_path="./cache" \
#    --eval_batch_size=16 \
#    --max_source_length=512 \
#    --save_name="Python800-CodeT5-large-test"\
#    --seed 123456