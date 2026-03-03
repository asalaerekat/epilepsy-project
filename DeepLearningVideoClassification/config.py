import argparse

def get_args():
    parser = argparse.ArgumentParser(description="3D Model Fine-Tuning Pipeline for Seizure Prediction")
    # Data parameters
    parser.add_argument("--video_dir", type=str, default="data/videos/",
                        help="Directory where internal videos are stored")
    parser.add_argument("--external_video_dir", type=str, default=None,
                        help="Directory where external evaluation videos are stored")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames per clip")
    parser.add_argument("--frame_rate", type=int, default=4,
                        help="Frame sampling rate")
    parser.add_argument("--split_mode", type=str, default="kfold",
                        choices=["kfold", "final_train", "external_eval"],
                        help="kfold: 5-fold CV on internal data; final_train: train on all internal data; external_eval: evaluate on external data only")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="resnet3d",
                        choices=["resnet3d", "vmz"], help="Model architecture to use")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross validation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--output_dir", type=str, default="checkpoints/", help="Directory to save models")
    parser.add_argument("--plot_dir", type=str, default="plots/", help="Directory to save loss plots")
    parser.add_argument("--best_fold", type=int, default=1, help="Pick best cv fold to start the finetuning on")
    parser.add_argument("--eval_only", action="store_true",
                        help="Deprecated: use --split_mode external_eval")
    args = parser.parse_args()
    return args
