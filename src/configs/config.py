import yaml
import os


# ============================================
# Configuration
# ============================================
class Config:
    # Objective
    training_objective: str = "mlfm"

    # Dataset
    data_path: str = None
    eval_data_path: str = None
    data_paths: list = None
    eval_data_paths: list = None
    data_mix_weights: list = None
    tokenized_data_root: str = "./data/mlfm"
    max_length: int = 128
    random_length_prob: float = 0.0  # Pretraining only: probability of cropping a full batch to a random prefix length.
    random_length_min: int = 1  # Minimum prefix length for random-length pretraining crops.
    random_length_max: int = 0  # 0 means use the current batch sequence length.

    # Training (optimizer + schedule)
    warmup_steps: int = 5000
    batch_size: int = None
    global_batch_size: int = 512
    lr: float = None
    base_lr: float = None
    weight_decay: float = 0.0
    base_weight_decay: float = None
    optimizer: str = "muon"  # "adamw" or "muon"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1  # Gradient accumulation steps (optimizer updates every K mini-batches)

    # EMA
    ema_decay1: float = 0.9999
    use_ema: bool = True
    eval_use_ema: bool = True

    # PPL Evaluation
    online_eval: bool = True  # Enable PPL evaluation for generated samples
    eval_ppl_model: str = "gpt2-large"  # Model for PPL evaluation
    eval_ppl_batch_size: int = 64  # Batch size for PPL evaluation (adjusted to be divisible by device count)
    eval_ppl_max_length: int = 1024  # Max sequence length for PPL evaluation
    generation_ppl_sample_count: int = 64  # Max decoded validation generations scored by eval_ppl_model.

    # Logging & Checkpointing
    log_freq: int = 100
    eval_freq: int = 10
    save_freq: float = 100  # Can be fractional (e.g., 0.1 for saving every 0.1 epoch)

    # Output
    output_dir: str = "./output_dir"
    resume: str = None
    reset_resume_training_state: bool = False  # Load weights from resume but start optimizer/step/RNG state from scratch.
    resume_adapter_weight_source: str = "model"  # Which resume adapter weights to load: "model" or "ema".
    restore_train_iterator_state: bool = False  # Exact dataloader offset replay on resume; false starts at next sampler epoch.
    resume_train_iterator_max_skip_batches: int = 2048  # Safety cap for exact replay; <0 means no cap.

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "MLFM"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_group: str = None
    wandb_job_type: str = None
    wandb_tag: str = None

    # Misc
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"
    precision: str = "bf16"  # "bf16", "fp16", or "fp32"
    compile: bool = False

    # MLFM backbone/adaptation
    backbone_model_name_or_path: str = None
    backbone_type: str = "auto"  # "llada", "smdm", or "auto"
    tokenizer_name_or_path: str = None
    trust_remote_code: bool = True
    model_loader: str = None  # "hf" or "smdm_litgpt"; inferred from backbone_type when omitted
    smdm_checkpoint_path: str = None
    smdm_code_path: str = None
    smdm_size: int = 1028
    backbone_hidden_dim: int = 0  # Optional sanity check; 0 means infer from model norm weights.
    mask_token_id: int = None
    freeze_backbone: bool = True
    freeze_embeddings: bool = True
    gradient_checkpointing: bool = True

    # Prompt/response SFT stage
    training_stage: str = "pretrain"  # "pretrain" or "sft"; sft keeps prompts clean and trains response tokens.
    sft_mix_weight: float = 0.0  # Optional packed+SFT microbatch mixing outside dedicated SFT stage.
    sft_total_tokens: int = 5242880000
    sft_max_steps: int = 10000
    sft_max_length: int = None  # Defaults to max_length.
    sft_batching_mode: str = "mixed_concat"  # "mixed_concat" or "source_schedule".
    sft_source_schedule_slots: int = 50
    sft_dynamic_crop: bool = False
    sft_dynamic_crop_multiple: int = 64
    sft_source_weights: dict = {"general": 0.35, "math": 0.45, "code": 0.2}
    sft_math_source_weights: dict = {"gsm8k_aug": 0.0, "gsm8k_aug_nl": 0.0, "metamathqa": 0.5, "math_reasoning": 0.5}
    sft_full_response_mask_prob: float = 0.5
    sft_general_data_paths: list = None
    sft_math_data_paths: list = None
    sft_math_gsm8k_data_paths: list = None
    sft_math_gsm8k_nl_data_paths: list = None
    sft_math_metamathqa_data_paths: list = None
    sft_code_data_paths: list = None
    sft_eval_data_paths: list = None

    # LoRA / AdaLN adapters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_freeze_base: bool = True
    lora_target_modules: list = None
    adaln: bool = True
    adaln_mode: str = "vanilla"  # "dit" -> block DiT adapter; "vanilla" -> norm-site shift/scale adapter.
    adaln_time_embed_dim: int = 256
    adaln_hidden_dim: int = 0  # Middle timestep MLP width; 0 means use each wrapped norm's hidden size.
    adaln_include_patterns: list = None
    adaln_exclude_patterns: list = None
    lora_output_head: bool = False
    lora_output_rank: int = 8
    lora_output_alpha: float = 16.0
    lora_output_dropout: float = 0.0
    lora_output_tied_policy: str = "tied_delta"  # "output_delta", "tied_delta", "untie_forbidden", or "allow_untie"

    # MLFM objective
    mask_ratio_sampler: str = "maskgit_cosine"  # "maskgit_cosine" or "uniform"
    maskgit_cosine_power: float = 1.0  # >1 shifts maskgit_cosine toward smaller mask ratios; <1 toward larger ratios.
    use_low_discrepancy: bool = False  # Stratify training mask_p/gamma samples by batch quantiles.
    mask_p_min: float = 0.05
    mask_p_max: float = 1.0
    mask_guarantee_nonempty: bool = True
    mlfm_loss_weighting: str = "inverse_count"  # "inverse_count", "inverse_p", or "none"
    lambda_mse: float = 0.0  # Weight for E[z0 | zt] embedding MSE on corrupted positions; 0 disables it.
    noise_parameterization: str = "log_nsr"
    forward_process: str = "brownian_bridge"
    brownian_bridge_sigma: float = 1.0  # Bridge noise scale and logNSR-to-bridge-t conversion scale.
    bridge_noise_covariance: str = "isotropic"  # "isotropic" or "empirical_diag" Brownian bridge noise geometry.
    bridge_noise_diag_max_tokens: int = 262144  # Global training-token budget used to estimate empirical_diag.
    bridge_noise_diag_min_tokens: int = 8192  # Use isotropic noise until at least this many tokens are accumulated.
    bridge_noise_diag_shrinkage: float = 0.05  # Shrink empirical diagonal variance toward its mean.
    bridge_noise_diag_eps: float = 1e-8  # Numerical floor for empirical diagonal scaling.
    bridge_noise_rank: int = 0  # Reserved for future low-rank covariance modes; ignored for empirical_diag.
    gamma_schedule: str = "gumbel"  # "uniform", "normal", "gumbel", "active_piecewise", or "active_mixture"
    gamma_min: float = -6.0
    gamma_max: float = 6.0
    gamma_loc: float = 0.0
    gamma_scale: float = 2.0
    gamma_active_piecewise_gamma: list = None  # EMA-smoothed active inverse-CDF gamma knots used by active schedules.
    gamma_active_piecewise_cdf: list = None  # CDF knots paired with gamma_active_piecewise_gamma.
    gamma_active_mixture_weights: list = None  # [uniform, normal, active_curve]; defaults to [0.1, 0.2, 0.7].
    gamma_curve_adapt_enabled: bool = False  # Estimate candidate gamma curve from diagnostics and EMA-update active curve.
    gamma_curve_estimator: str = "generalized_logistic"  # "generalized_logistic", "normal", or "empirical".
    gamma_curve_shape_min: float = 0.05
    gamma_curve_shape_max: float = 20.0
    gamma_curve_update_rate: float = 0.02
    gamma_curve_quantile_points: int = 101
    gamma_curve_mask_p_min: float = 0.05
    gamma_curve_mask_p_max: float = 1.0
    gamma_curve_min_bins: int = 8
    gamma_curve_min_examples: int = 4096
    gamma_curve_smoothing: str = "isotonic"
    gamma_curve_min_r2: float = 0.95
    gamma_curve_updates: int = 0
    gamma_curve_last_update_step: int = 0
    restore_adaptive_gamma_state: bool = True
    restore_loss_diagnostic_state: bool = True
    loss_diagnostic_estimator: str = "window"  # "window" preserves old rolling-buffer behavior; "ema" stores decayed bin stats.
    loss_diagnostic_ema_decay: float = 0.98  # Per-log decay for loss_diagnostic_estimator=ema.
    time_conditioning: str = "gamma"
    special_token_ids: list = None

    # MLFM training defaults
    max_train_steps: int = 100000
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    lora_lr: float = 1e-4
    adaln_lr: float = 3e-4
    lora_output_lr: float = 1e-4
    save_full_model: bool = False
    eval_max_batches: int = 32
    run_generation_validation: bool = False
    val_num_generation_samples: int = 64
    generation_sampler: str = "sde"  # Camera-ready validation generation uses stochastic Brownian-bridge steps.
    val_generation_steps: int = 0  # 0 means min(sequence length, val_generation_steps_cap).
    val_generation_steps_cap: int = 128
    wandb_generation_sample_count: int = 8
    wandb_generation_max_chars: int = 2000
    val_sft_prompt_generation_samples_per_type: int = 2
    val_sft_prompt_generation_max_batches: int = 16
    sample_diagnostics: bool = True
    sample_diagnostics_window_batches: int = 100
    sample_diagnostics_update_every_batches: int = 20
    loss_diagnostics_gamma_bins: int = 0  # 0 derives gamma bins from the estimator's effective sample count.
    loss_diagnostics_target_samples_per_cell: int = 200
    loss_diagnostics_mask_bin_width: float = 0.05
    embedding_geometry_empirical_tokens: int = 65536  # Max training tokens per rank used for embedding covariance diagnostics.
    embedding_geometry_empirical_batches: int = 64  # Max dataloader batches per rank used for embedding covariance diagnostics.
    val_unconditional_generation_samples: int = 8
    val_gsm8k_generation_samples: int = 8
    gsm8k_eval_path: str = "data/gsm8k/test.jsonl"


def load_config_from_yaml(path: str) -> Config:
    """Load a YAML config and override defaults in Config."""
    config = Config()
    if not path or not os.path.isfile(path):
        return config

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    for key, value in cfg_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def apply_config_overrides(config: Config, overrides: list) -> Config:
    """Apply command-line config overrides to a Config object.
    
    Args:
        config: Config object to modify
        overrides: List of strings in format "field_name=value"
    
    Returns:
        Modified config object
    """
    if not overrides:
        return config
    
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'field_name=value'")
        
        field_name, value_str = override.split("=", 1)
        field_name = field_name.strip()
        value_str = value_str.strip()
        
        if not hasattr(config, field_name):
            raise ValueError(f"Config has no field named '{field_name}'")
        
        original_value = getattr(config, field_name)
        original_type = type(original_value)

        # Allow setting a field back to None
        if value_str.lower() == "none":
            setattr(config, field_name, None)
            continue

        if original_value is None:
            # Use type annotation to infer the intended type
            annotated_type = config.__annotations__.get(field_name)
            if annotated_type == int:
                converted_value = int(value_str)
            elif annotated_type == float:
                converted_value = float(value_str)
            elif annotated_type == bool:
                converted_value = value_str.lower() in ("true", "1", "yes")
            elif annotated_type == list:
                converted_value = yaml.safe_load(value_str)
                if converted_value is not None and not isinstance(converted_value, list):
                    converted_value = [converted_value]
            else:
                converted_value = value_str
        elif original_type == bool:
            converted_value = value_str.lower() in ("true", "1", "yes")
        elif original_type == int:
            converted_value = int(value_str)
        elif original_type == float:
            converted_value = float(value_str)
        elif original_type == str:
            converted_value = value_str
        elif original_type == list:
            converted_value = yaml.safe_load(value_str)
            if converted_value is not None and not isinstance(converted_value, list):
                converted_value = [converted_value]
        elif original_type == dict:
            converted_value = yaml.safe_load(value_str)
            if not isinstance(converted_value, dict):
                raise ValueError(f"Expected a dict override for '{field_name}', got {type(converted_value).__name__}")
        else:
            converted_value = value_str
        
        setattr(config, field_name, converted_value)

    return config
