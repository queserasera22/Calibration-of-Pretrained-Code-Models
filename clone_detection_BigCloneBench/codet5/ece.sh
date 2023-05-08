CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/checkpoint-best-f1/pytorch_model.bin" \
    --model_type codet5 \
    --task clone \
    --test_filename="../dataset/test_sampled.txt" \
    --cache_path="./cache" \
    --eval_batch_size=16 \
    --max_source_length=512 \
    --save_name="Clone-CodeT5"\
    --seed 123456


#CUDA_VISIBLE_DEVICES=2 python ece.py \
#    --model_name_or_path='Salesforce/codet5-small' \
#    --config_name='Salesforce/codet5-small' \
#    --tokenizer_name='Salesforce/codet5-small' \
#    --load_model_path="models/codet5_small/checkpoint-best-f1/pytorch_model.bin" \
#    --model_type codet5 \
#    --task clone \
#    --test_filename="../dataset/test_sampled.txt" \
#    --cache_path="./cache" \
#    --eval_batch_size=8 \
#    --max_source_length=512 \
#    --save_name="Clone-CodeT5-small"\
#    --seed 123456

#CUDA_VISIBLE_DEVICES=3 python ece.py \
#    --model_name_or_path='Salesforce/codet5-large' \
#    --config_name='Salesforce/codet5-large' \
#    --tokenizer_name='Salesforce/codet5-large' \
#    --load_model_path="models/codet5_large/checkpoint-best-f1/pytorch_model.bin" \
#    --model_type codet5 \
#    --task clone \
#    --test_filename="../dataset/test_sampled.txt" \
#    --cache_path="./cache" \
#    --eval_batch_size=2 \
#    --max_source_length=512 \
#    --save_name="Clone-CodeT5-large"\
#    --seed 123456


CUDA_VISIBLE_DEVICES=3 python ece.py \
    --model_name_or_path='Salesforce/codet5-base' \
    --config_name='Salesforce/codet5-base' \
    --tokenizer_name='Salesforce/codet5-base' \
    --load_model_path="models/codet5_base_ls_03/checkpoint-best-f1/pytorch_model.bin" \
    --model_type codet5 \
    --task clone \
    --test_filename="../dataset/test_sampled.txt" \
    --cache_path="./cache" \
    --eval_batch_size=8 \
    --max_source_length=512 \
    --save_name="Clone-CodeT5-LS-03"\
    --seed 123456