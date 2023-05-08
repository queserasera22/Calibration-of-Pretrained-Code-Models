CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5/checkpoint-best-acc/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --test_filename="../dataset/test.jsonl" \
    --num_labels 20 \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Exception-CodeT5"\
    --seed 123456

#
#CUDA_VISIBLE_DEVICES=3 python ece.py \
#    --model_name_or_path='Salesforce/codet5-large' \
#    --config_name='Salesforce/codet5-large' \
#    --tokenizer_name='Salesforce/codet5-large' \
#    --load_model_path="models/codet5_large/checkpoint-best-acc/pytorch_model.bin" \
#    --model_type codet5 \
#    --task classification \
#    --test_filename="../dataset/test.jsonl" \
#    --num_labels 20 \
#    --cache_path="./cache" \
#    --eval_batch_size=16 \
#    --max_source_length=512 \
#    --save_name="Exception-CodeT5-large"\
#    --seed 123456


CUDA_VISIBLE_DEVICES=1 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5/checkpoint-best-acc/ts/model.pt" \
    --model_type codet5 \
    --task classification \
    --test_filename="../dataset/test.jsonl" \
    --num_labels 20 \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Exception-CodeT5-TS"\
    --seed 123456

CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name_or_path='Salesforce/codet5-large' \
    --config_name='Salesforce/codet5-large' \
    --tokenizer_name='Salesforce/codet5-large' \
    --load_model_path="models/codet5_large_test/checkpoint-best-acc/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --test_filename="../dataset/test.jsonl" \
    --num_labels 20 \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Exception-CodeT5-large"\
    --seed 123456