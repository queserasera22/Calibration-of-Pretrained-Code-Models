CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/checkpoint-best-acc/pytorch_model.bin" \
    --model_type codet5 \
    --task defect \
    --test_filename="../dataset/test.jsonl" \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Defect-CodeT5"\
    --seed 123456

CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name_or_path='Salesforce/codet5-large' \
    --config_name='Salesforce/codet5-large' \
    --tokenizer_name='Salesforce/codet5-large' \
    --load_model_path="models/codet5_large_test/checkpoint-best-acc/pytorch_model.bin" \
    --model_type codet5 \
    --task defect \
    --test_filename="../dataset/test.jsonl" \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Defect-CodeT5-large"\
    --seed 123456

CUDA_VISIBLE_DEVICES=1 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5_base_ls_03/checkpoint-best-acc/pytorch_model.bin" \
    --model_type codet5 \
    --task defect \
    --test_filename="../dataset/test.jsonl" \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Defect-CodeT5-LS-03"\
    --seed 123456


CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5_base_test/checkpoint-best-acc/pytorch_model.bin" \
    --model_type codet5 \
    --task defect \
    --test_filename="../dataset/test.jsonl" \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Defect-CodeT5-base-test"\
    --seed 123456