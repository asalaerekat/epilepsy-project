import argparse

def get_args():
    parser = argparse.ArgumentParser(description="3D Model Fine-Tuning Pipeline for Seizure Prediction")
    parser.add_argument("--run_mode", type=str, default="train",
                        choices=["train", "test_only"],
                        help="train: run grouped K-fold CV then final training and external eval; test_only: evaluate a saved final model on external data")

    # Data parameters
    parser.add_argument("--video_dir", type=str, default="data/videos/",
                        help="Directory where internal videos are stored")
    parser.add_argument("--external_video_dir", type=str, default=None,
                        help="Directory where external test/evaluation videos are stored")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames per clip")
    parser.add_argument("--frame_rate", type=int, default=4,
                        help="Frame sampling rate")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="resnet3d",
                        choices=["resnet3d", "vmz"], help="Model architecture to use")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--eval_num_workers", type=int, default=0,
                        help="Number of workers for external evaluation DataLoader (0 avoids worker deadlocks)")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Batch size for external evaluation (smaller values reduce OOM risk on long videos)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for classifier head (fc)")
    parser.add_argument("--layer4_learning_rate", type=float, default=1e-4,
                        help="Learning rate for backbone layer4")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross validation")
    parser.add_argument("--scheduler", type=str, default="step",
                        choices=["step", "cosine"], help="Learning rate scheduler type")
    parser.add_argument("--lr_step_size", type=int, default=10,
                        help="StepLR step size (used when --scheduler step)")
    parser.add_argument("--lr_gamma", type=float, default=0.5,
                        help="StepLR gamma (used when --scheduler step)")
    parser.add_argument("--lr_t_max", type=int, default=0,
                        help="CosineAnnealingLR T_max (0 means use --epochs)")
    parser.add_argument("--lr_eta_min", type=float, default=1e-6,
                        help="CosineAnnealingLR eta_min (used when --scheduler cosine)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--output_dir", type=str, default="checkpoints/", help="Directory to save models")
    parser.add_argument("--plot_dir", type=str, default="plots/", help="Directory to save loss plots")
    parser.add_argument("--use_avg_youden_threshold", action="store_true",
                        help="Use mean K-fold Youden threshold for final/external evaluation. If not set, uses 0.5.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Checkpoint path for test_only mode. Defaults to <output_dir>/final_model.pth")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Explicit threshold for test_only mode. If omitted, loads final_threshold.json or uses 0.5")
    
    # this is for different sampling techniques
    parser.add_argument("--sample_mode", type=str, default="random_clip",
                    choices=["sparse", "random_clip", "center_clip"],
                    help="How to sample frames from each video.")

    parser.add_argument("--clip_fps", type=float, default=4.0,
                        help="Effective frame rate inside each sampled clip.")

    parser.add_argument("--num_test_clips", type=int, default=20,
                        help="Number of deterministic clips per video during multi-clip testing.")

    parser.add_argument("--aggregation", type=str, default="topk_mean",
                        choices=["mean", "max", "topk_mean"],
                        help="How to aggregate clip probabilities into a video probability.")

    parser.add_argument("--topk", type=int, default=5,
                        help="Top-k clips used when aggregation=topk_mean.")
        
    args = parser.parse_args()
    return args
