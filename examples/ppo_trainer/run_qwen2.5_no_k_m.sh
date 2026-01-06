set -x

# hotpot_qa_train_path=/mnt/tidal-alsh01/usr/chenyiqun/datasets/data/verl_format_data/hotpotqa/train_verl.parquet
# hotpot_qa_test_path=/mnt/tidal-alsh01/usr/chenyiqun/datasets/data/verl_format_data/hotpotqa/test_verl.parquet
# train_files="['$hotpot_qa_train_path']"
# test_files="['$hotpot_qa_test_path']"

nq_search_train_path=/mnt/tidal-alsh01/usr/chenyiqun/datasets/data/verl_format_data/nq_search/train_verl.parquet
nq_search_test_path=/mnt/tidal-alsh01/usr/chenyiqun/datasets/data/verl_format_data/nq_search/test_verl.parquet
train_files="['$nq_search_train_path']"
test_files="['$nq_search_test_path']"

# WandB 登录
export WANDB_API_KEY="5235f681e1a2a0ef6fe3a1f4686280daad738532"
# vllm
export VLLM_USE_V1=1

echo ' ***************** export vllm v1 ***************** '
echo ' ***************** export wandb api_key ***************** '

python3 -m verl.trainer.main_ppo_no_k_m \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.val_batch_size=1024 \
    data.max_prompt_length=3072 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=2\
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.project_name='knowledge_search_nq' \
    trainer.experiment_name='No_Knowledge_Agents_maxiter_5' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=15 $@ \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.actor.ppo_epochs=1 \
    trainer.val_before_train=True \

# actor_rollout_ref.rollout.temperature=0.7 \
# actor_rollout_ref.rollout.top_p=0.8 \

# actor_rollout_ref.actor.use_kl_loss=True \
# actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
