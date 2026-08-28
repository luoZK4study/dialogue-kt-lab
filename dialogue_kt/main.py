import argparse

from dialogue_kt.annotate import annotate
from dialogue_kt.human_eval import human_eval
from dialogue_kt.training import train, test, BASELINE_MODELS
from dialogue_kt.visualize import visualize
from dialogue_kt.utils import initialize_seeds, bool_type

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_annotate = subparsers.add_parser("annotate", help="Annotate dialogues")
    parser_annotate.set_defaults(func=annotate)
    parser_annotate.add_argument("--mode", type=str, choices=["collect", "analyze"], help="Collect annotations or analyze existing files")
    parser_annotate.add_argument("--use_azure", action="store_true", help="Use Azure endpoint")
    parser_annotate.add_argument("--openai_model", type=str, help="Model identifier string")

    parser_hum_eval = subparsers.add_parser("human-eval", help="Create/analyze human evaluation files")
    parser_hum_eval.set_defaults(func=human_eval)
    parser_hum_eval.add_argument("--mode", type=str, choices=["create", "analyze"], help="Create evaluation files or analyze existing files")

    parser_train = subparsers.add_parser("train", help="Train KT model")
    parser_train.set_defaults(func=train)
    parser_train.add_argument("--epochs", type=int, help="Deprecated compatibility alias for --max_epochs")
    parser_train.add_argument("--max_epochs", "--max-epochs", dest="max_epochs", type=int,
                              help="Maximum number of end-to-end training epochs")
    parser_train.add_argument("--patience", type=int, default=0,
                              help="Early-stopping patience in validation epochs; 0 disables early stopping")
    parser_train.add_argument("--min_delta", "--min-delta", dest="min_delta", type=float, default=0.0,
                              help="Minimum validation-loss decrease required to reset early-stopping patience")
    parser_train.add_argument("--lr", type=float, help="Learning rate")
    parser_train.add_argument("--cel_calibrator_lr", type=float,
                              help="Optional dedicated learning rate for the CEL output calibrator; defaults to --lr when unset")
    parser_train.add_argument("--cel_selector_lr", type=float,
                              help="Optional dedicated learning rate for A during joint CEL training")
    parser_train.add_argument("--cel_environment_lr", type=float,
                              help="Optional dedicated learning rate for B during joint A+B dual-path training")
    parser_train.add_argument("--cel_calibrator_warmup_epochs", type=int, default=0,
                              help="Optional number of initial epochs that train only the CEL output calibrator before restoring the normal CEL train mode")
    parser_train.add_argument("--cel_calibrator_warmup_lr", type=float,
                              help="Optional dedicated learning rate used only during CEL calibrator warmup epochs; defaults to the active calibrator/base lr when unset")
    parser_train.add_argument("--wd", type=float, help="Weight decay")
    parser_train.add_argument("--gc", type=float, help="Gradient clipping norm")
    parser_train.add_argument("--grad_accum_steps", type=int, help="Steps to accumulate gradients for")
    parser_train.add_argument("--r", type=int, help="LoRA rank")
    parser_train.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser_train.add_argument("--model_init_seed", type=int, default=221,
                              help="Seed used immediately before model, LoRA, selector, and environment initialization")
    parser_train.add_argument("--optim", type=str, choices=["adamw", "adafactor"], default="adamw", help="Optimizer")
    parser_train.add_argument("--pt_model_name", type=str, help="Name of pre-trained model to initialize weights from")
    parser_train.add_argument("--cel_selector_init_model_name", type=str,
                              help="Optional CEL model checkpoint used to warm-start selector weights only")
    parser_train.add_argument("--cel_calibrator_init_model_name", type=str,
                              help="Optional CEL model checkpoint used to warm-start calibrator weights only")
    parser_train.add_argument("--cel_environment_init_model_name", type=str,
                              help="Optional same-candidate B checkpoint used to initialize the environment generator")
    parser_train.add_argument("--cel_require_exact_selector_init", type=bool_type, default=False,
                              help="Fail instead of falling back to a fresh or partial selector when an init checkpoint is requested")
    parser_train.add_argument("--cel_require_complete_checkpoint", type=bool_type, default=False,
                              help="Require selector and calibrator artifacts when testing the trained CEL checkpoint")
    parser_train.add_argument("--cel_require_exact_environment_init", type=bool_type, default=False,
                              help="Require the B checkpoint to exactly match the requested generator")
    parser_train.add_argument("--skip_test_after_train", type=bool_type, default=False,
                              help="Stop after training and validation checkpoint selection for an intermediate experiment stage")
    parser_train.add_argument("--cel_train_environment_only", type=bool_type, default=False,
                              help="Historical special protocol: freeze A, LoRA, and calibrator while warming up B")
    parser_train.add_argument("--hyperparam_sweep", action="store_true", help="Run a hyperparameter sweep experiment")

    parser_test = subparsers.add_parser("test", help="Test KT model")
    parser_test.set_defaults(func=test)
    parser_test.add_argument("--cel_require_complete_checkpoint", type=bool_type, default=False,
                             help="Require selector and calibrator artifacts when testing the trained CEL checkpoint")

    parser_visualize = subparsers.add_parser("visualize", help="Visualize KCs")
    parser_visualize.set_defaults(func=visualize)

    for subparser in [parser_annotate, parser_hum_eval, parser_train, parser_test, parser_visualize]:
        subparser.add_argument("--dataset", type=str, choices=["comta", "mathdial"], default="comta", help="Which dataset to use")
        subparser.add_argument("--split_by_subject", action="store_true", help="For CoMTA, define train/test and folds using subjects")
        subparser.add_argument("--typical_cutoff", type=int, default=1, help="For MathDial, lowest acceptable dialogue 'typical' score")
        subparser.add_argument("--tag_src", type=str, choices=["base", "atc"], default="atc", help="Source of KC tags - base: generated by LLM, atc: ATC standards")
        subparser.add_argument("--debug", action="store_true", help="Use subset of data for debugging")

    for subparser in [parser_train, parser_test, parser_visualize]:
        subparser.add_argument("--model_type", type=str, choices=["lmkt", "random", "majority", "bkt"] + BASELINE_MODELS, default="lmkt", help="Model architecture to use")
        subparser.add_argument("--model_name", type=str, help="Name of model to save for training or load for testing")
        subparser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B", help="HuggingFace base model for LLMKT")
        subparser.add_argument("--inc_first_label", action="store_true", help="Include first turn label in dialogues when testing")

    for subparser in [parser_train, parser_test]:
        subparser.add_argument("--batch_size", type=int, help="Model batch size")
        subparser.add_argument("--crossval", action="store_true", help="Run training/testing over all folds and aggregate results")
        subparser.add_argument("--testonval", action="store_true", help="Run testing phase on validation set (automatic for hyperparam_sweep)")
        subparser.add_argument("--agg", type=str, choices=["prod", "mean-ar", "mean-geo"], default="mean-ar", help="Method for aggregating KC probabilities into correctness probability")
        subparser.add_argument("--result_subdir", type=str, default=None, help="Optional results subdirectory under results/")
        subparser.add_argument("--pack_kcs", type=bool_type, default=True, help="For LLMKT, pack all KCs for a turn in a single prompt")
        subparser.add_argument("--quantize", type=bool_type, default=True, help="Quantize LLMKT base model")
        subparser.add_argument("--prompt_inc_labels", type=bool_type, default=False, help="For LLMKT, include explicit correctness and KC labels in prompt")
        subparser.add_argument("--emb_size", type=int, help="Latent state dimension for DKT family models")
        subparser.add_argument("--cel_mode", type=str, choices=["mlp", "adapter", "task_conditioned"], default=None,
                               help="CEL A selector mode: mlp, adapter, or task_conditioned")
        subparser.add_argument("--cel_layer_idx", type=int, default=-1, help="Layer index used to extract H_k for CEL injection when `--cel_hook_site last_block`; -1 means the last transformer layer")
        subparser.add_argument("--cel_hook_site", type=str, choices=["last_block", "pre_lm_head"], default="last_block",
                               help="Where to hook CEL injection: last transformer block output or final hidden state right before lm_head")
        subparser.add_argument("--cel_hook_timing", type=str, choices=["post_block", "pre_block"], default="post_block",
                               help="Apply last-block CEL injection after the selected block or to its input before self-attention")
        subparser.add_argument("--cel_gamma", type=float, default=0.3, help="Residual rationale injection strength")
        subparser.add_argument("--cel_selector_hidden_dim", type=int, default=None, help="Hidden size for CEL selector MLP/adapter")
        subparser.add_argument("--cel_adapter_dim", type=int, default=None, help="Bottleneck size for CEL adapter selector")
        subparser.add_argument("--cel_drop", type=float, default=0.1, help="Dropout for CEL selector")
        subparser.add_argument("--cel_use_norm", type=bool_type, default=False, help="Apply extra RMS norm after rationale injection")
        subparser.add_argument("--cel_injection_variant", type=str, choices=["scalar_gate", "vector_shift"], default="scalar_gate",
                               help="CEL residual parameterization: scalar gate scaling or vector shift injection")
        subparser.add_argument("--cel_application_mode", type=str, choices=["token_residual", "prediction_token_shift"], default="token_residual",
                               help="Where the CEL residual is applied: rewrite context tokens or add a pooled evidence shift only at prediction tokens")
        subparser.add_argument("--cel_train_selector_only", type=bool_type, default=False,
                               help="Freeze loaded LoRA/backbone parameters and train only the CEL selector")
        subparser.add_argument("--cel_output_calibration", type=str, choices=["none", "bias", "affine"], default="none",
                               help="Optional monotonic output calibrator applied to final corr_probs: none, logit bias, or positive affine logit transform")
        subparser.add_argument("--cel_calibrator_init_bias", type=float, default=0.0,
                               help="Optional initial logit bias for the CEL output calibrator; useful for warm-starting threshold correction")
        subparser.add_argument("--cel_calibrator_init_scale", type=float, default=1.0,
                               help="Optional initial positive logit scale for affine CEL calibrators; 1.0 keeps the initial transform as identity")
        subparser.add_argument("--cel_train_calibrator_only", type=bool_type, default=False,
                               help="Freeze backbone and selector, and train only the CEL output calibrator")
        subparser.add_argument("--cel_train_selector_and_calibrator_only", type=bool_type, default=False,
                               help="Freeze backbone/LoRA parameters and jointly train only the CEL selector plus optional output calibrator")
        subparser.add_argument("--cel_stage2_enabled", type=bool_type, default=False,
                               help="Enable the B/non-evidence branch for joint A+B dual-path training")
        subparser.add_argument("--cel_stage2_fresh_init", type=bool_type, default=False,
                               help="Require an A+B candidate to start from the raw base model")
        subparser.add_argument("--cel_stage2_candidate_id", type=str, default=None,
                               help="Stable identifier for one independently initialized A+B candidate")
        subparser.add_argument("--cel_stage2_phase", type=str,
                               choices=["a_bootstrap", "calibrator_warmup", "a_joint", "b_warmup", "joint"], default=None,
                               help="Historical compatibility phase; leave unset for unified A+B training")
        subparser.add_argument("--cel_stage2_parent_model_name", type=str, default=None,
                               help="Historical same-candidate parent checkpoint for a staged compatibility phase")
        subparser.add_argument("--cel_env_mode", type=str, choices=["shuffle", "mlp", "transformer", "contextual_transformer"], default=None,
                               help="B module direction")
        subparser.add_argument("--cel_env_beta", type=float, default=0.1,
                               help="Environment residual scale before the shared CEL gamma")
        subparser.add_argument("--cel_env_split_mode", type=str, choices=["complementary", "topk_abs", "sigmoid"], default="complementary",
                               help="Convert selector output into the rationale/non-rationale split")
        subparser.add_argument("--cel_env_topk_ratio", type=float, default=0.1,
                               help="Fraction of context tokens assigned to the rationale split for topk_abs")
        subparser.add_argument("--cel_env_sigmoid_temperature", type=float, default=5.0,
                               help="Temperature used by the optional sigmoid split ablation")
        subparser.add_argument("--cel_env_hidden_dim", type=int, default=1024,
                               help="Internal width of the learned B module")
        subparser.add_argument("--cel_env_num_layers", type=int, default=4,
                               help="Number of Transformer encoder layers in the B Transformer")
        subparser.add_argument("--cel_env_num_heads", type=int, default=8,
                               help="Number of attention heads in the B Transformer")
        subparser.add_argument("--cel_env_ffn_dim", type=int, default=4096,
                               help="Optional Transformer environment FFN width")
        subparser.add_argument("--cel_env_drop", type=float, default=0.1,
                               help="Dropout inside the learned B module")
        subparser.add_argument("--cel_env_output_postprocess", type=str, choices=["none", "centered_rms"], default="centered_rms",
                               help="Optional post-processing that centers learned environment tokens and calibrates their RMS")
        subparser.add_argument("--cel_env_output_ratio", type=float, default=1.0,
                               help="Target environment-to-input RMS ratio used by centered_rms post-processing")
        subparser.add_argument("--cel_env_output_init_std", type=float, default=0.01,
                               help="Output projection initialization standard deviation for the Transformer environment generator")
        subparser.add_argument("--cel_env_shuffle_seed", type=int, default=221,
                               help="Seed for deterministic within-context non-rationale shuffling")
        subparser.add_argument("--cel_stage2_lambda_r", type=float, default=1.0,
                               help="Evidence-path BCE weight in the joint A+B objective")
        subparser.add_argument("--cel_stage2_lambda_m", type=float, default=1.0,
                               help="Mixed-path BCE weight in the joint A+B objective")
        subparser.add_argument("--cel_stage2_lambda_cons", type=float, default=0.1,
                               help="Bernoulli Jensen-Shannon consistency weight")
        subparser.add_argument("--cel_stage2_beta_start_ratio", type=float, default=0.2,
                               help="Initial fraction of target beta during B warmup")
        subparser.add_argument("--cel_stage2_consistency_ramp_fraction", type=float, default=0.25,
                               help="Fraction of B warmup used to ramp the consistency weight")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
