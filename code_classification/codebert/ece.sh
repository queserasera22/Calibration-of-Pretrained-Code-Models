CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/checkpoint/POJ104/model.bin" \
    --test_data_file="../dataset/POJ104/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=104 \
    --save_name="POJ104-CodeBERT"

CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/checkpoint/Java250/model.bin" \
    --test_data_file="../dataset/CodeNet_Java250/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=250 \
    --save_name="Java250-CodeBERT"

CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert/checkpoint/Python800/model.bin" \
    --test_data_file="../dataset/CodeNet_Python800/test.jsonl" \
    --eval_batch_size=128 \
    --block_size=512 \
    --num_labels=800 \
    --save_name="Python800-CodeBERT"


CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta/checkpoint/POJ104/model.bin" \
    --test_data_file="../dataset/POJ104/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=104 \
    --save_name="POJ104-CodeBERTa"

CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta/checkpoint/Java250/model.bin" \
    --test_data_file="../dataset/CodeNet_Java250/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=250 \
    --save_name="Java250-CodeBERTa"

CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta/checkpoint/Python800/model.bin" \
    --test_data_file="../dataset/CodeNet_Python800/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=800 \
    --save_name="Python800-CodeBERTa"


CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert_OOD_token/Java250/model.bin" \
    --test_data_file="../dataset/CodeNet_Java250/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=250 \
    --save_name="Java250-CodeBERT-OOD-token"


CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta_OOD_token/Java250/model.bin" \
    --test_data_file="../dataset/CodeNet_Java250/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=250 \
    --save_name="Java250-CodeBERTa-OOD-token"



CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="microsoft/codebert-base" \
    --model_path="models/codebert_OOD/Java250/model.bin" \
    --test_data_file="../dataset/CodeNet_Java250/OOD/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=250 \
    --save_name="Java250-CodeBERT-OOD"


CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="huggingface/CodeBERTa-small-v1" \
    --model_path="models/codeberta_OOD/Java250/model.bin" \
    --test_data_file="../dataset/CodeNet_Java250/OOD/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=250 \
    --save_name="Java250-CodeBERTa-OOD"

#CUDA_VISIBLE_DEVICES=0 python ece.py \
#    --model_name="microsoft/codebert-base" \
#    --model_path="models/codebert_OOD_token/Python800/model.bin" \
#    --test_data_file="../dataset/CodeNet_Python800/OOD_token/test.jsonl" \
#    --eval_batch_size=32 \
#    --block_size=512 \
#    --num_labels=800 \
#    --save_name="Python800-CodeBERT-OOD-token-test"


#CUDA_VISIBLE_DEVICES=0 python ece.py \
#    --model_name="huggingface/CodeBERTa-small-v1" \
#    --model_path="models/codeberta_OOD_token/Python800/model.bin" \
#    --test_data_file="../dataset/CodeNet_Python800/OOD_token/test.jsonl" \
#    --eval_batch_size=32 \
#    --block_size=512 \
#    --num_labels=800 \
#    --save_name="Python800-CodeBERTa-OOD-token"
#
#
#
#CUDA_VISIBLE_DEVICES=0 python ece.py \
#    --model_name="microsoft/codebert-base" \
#    --model_path="models/codebert_OOD/ls_01/Python800/model.bin" \
#    --test_data_file="../dataset/CodeNet_Python800/OOD/test.jsonl" \
#    --eval_batch_size=64 \
#    --block_size=512 \
#    --num_labels=250 \
#    --save_name="Python800-CodeBERT-OOD-LS-01"


CUDA_VISIBLE_DEVICES=0 python ece.py \
    --model_name="microsoft/graphcodebert-base" \
    --model_path="models/graphcodebert/POJ104/model.bin" \
    --test_data_file="../dataset/POJ104/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=104 \
    --save_name="POJ104-GraphCodeBERT"

CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name="microsoft/graphcodebert-base" \
    --model_path="models/graphcodebert_12345/Java250/model.bin" \
    --test_data_file="../dataset/CodeNet_Java250/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=250 \
    --save_name="Java250-GraphCodeBERT"

CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name="microsoft/graphcodebert-base" \
    --model_path="models/graphcodebert/Python800/model.bin" \
    --test_data_file="../dataset/CodeNet_Python800/test.jsonl" \
    --eval_batch_size=128 \
    --block_size=512 \
    --num_labels=800 \
    --save_name="Python800-GraphCodeBERT"

CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name="microsoft/graphcodebert-base" \
    --model_path="models/graphcodebert_OOD/Java250/model.bin" \
    --test_data_file="../dataset/CodeNet_Java250/OOD/test.jsonl" \
    --eval_batch_size=64 \
    --block_size=512 \
    --num_labels=250 \
    --save_name="Java250-GraphCodeBERT-OOD"

CUDA_VISIBLE_DEVICES=2 python ece.py \
    --model_name="microsoft/graphcodebert-base" \
    --model_path="models/graphcodebert_OOD_token/Python800/model.bin" \
    --test_data_file="../dataset/CodeNet_Python800/OOD_token/test.jsonl" \
    --eval_batch_size=128 \
    --block_size=512 \
    --num_labels=800 \
    --save_name="Python800-GraphCodeBERT-OOD-token"