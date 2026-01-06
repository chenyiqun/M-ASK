set -x

hotpot_qa_train_path=hotpot_qa_train_path
hotpot_qa_test_path=hotpot_qa_test_path
train_files="['$hotpot_qa_train_path']"
test_files="['$hotpot_qa_test_path']"

# WandB 登录
export WANDB_API_KEY="xxxxxxx"
# vllm
export VLLM_USE_V1=1

echo ' ***************** export vllm v1 ***************** '
echo ' ***************** export wandb api_key ***************** '

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=1024 \
    data.max_prompt_length=3072 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=model_path \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=model_path \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.project_name='knowledge_search_hotpotqa' \
    trainer.experiment_name='M-ASK' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=15 $@ \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.actor.ppo_epochs=1 \
    trainer.val_before_train=True \