#!/bin/bash
# Manual evaluation script for the trained UltraFeedback model
# Use this to evaluate models that were saved with the corrected path format

# The model was actually saved to:
MODEL_PATH="experiments/streaming_vpl_ultrafeedback/all_seq10_latent64_hidden256_beta0.12_freebits8.0_cycles4_gamma1.1_seed0/slurm_6905530/final_checkpoint/model.pt"

# Run evaluation
python -m hidden_context.evaluate_streaming_vpl \
    --model_path "$MODEL_PATH" \
    --data_path "data_release/P_4_survey_100/gpt2" \
    --data_subset "all" \
    --seq_length 10 \
    --latent_dim 64 \
    --hidden_dim 256 \
    --use_transformer \
    --num_attention_heads 4 \
    --num_transformer_layers 2 \
    --num_eval_episodes 500 \
    --seed 0

