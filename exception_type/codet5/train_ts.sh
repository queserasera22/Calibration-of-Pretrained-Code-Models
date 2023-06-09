CUDA_VISIBLE_DEVICES=2 python train_ts.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5/checkpoint-best-acc/pytorch_model.bin" \
    --model_type codet5 \
    --task classification \
    --eval_filename="../dataset/valid.jsonl" \
    --test_filename="../dataset/test.jsonl" \
    --num_labels 20 \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --seed 123456