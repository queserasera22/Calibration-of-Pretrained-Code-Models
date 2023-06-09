CUDA_VISIBLE_DEVICES=0 python train_ts.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/checkpoint-best-f1/pytorch_model.bin" \
    --model_type codet5 \
    --task clone \
    --eval_filename="../dataset/valid_sampled.txt" \
    --test_filename="../dataset/test_sampled.txt" \
    --cache_path="./cache" \
    --eval_batch_size=8 \
    --max_source_length=512 \
    --seed 123456