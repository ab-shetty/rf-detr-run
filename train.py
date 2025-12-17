import os
import zipfile
import argparse
import yaml
import wandb
from pathlib import Path
from rfdetr import RFDETRSmall

def extract_dataset(zip_path, extract_to="/tmp/dataset"):
  
    print(f"Extracting dataset from {zip_path}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Dataset extracted to {extract_to}")
    return extract_to

def restructure_dataset(extract_dir):
    """Reorganize dataset to RF-DETR format"""
    import shutil
    
    dataset_dir = os.path.join(extract_dir, 'dataset')
    
    # Rename val to valid
    val_dir = os.path.join(dataset_dir, 'val')
    if os.path.exists(val_dir):
        os.rename(val_dir, os.path.join(dataset_dir, 'valid'))
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        
        # Move images from images/ subfolder to split folder
        if os.path.exists(images_dir):
            for img in os.listdir(images_dir):
                shutil.move(os.path.join(images_dir, img), 
                           os.path.join(split_dir, img))
            os.rmdir(images_dir)
        
        # Move annotation file
        anno_src = f'annotations_{split if split != "valid" else "val"}.json'
        anno_path = os.path.join(dataset_dir, anno_src)
        if os.path.exists(anno_path):
            shutil.move(anno_path, 
                       os.path.join(split_dir, '_annotations.coco.json'))
    
    return dataset_dir
  
def train_rfdetr(args):
    # Initialize Weights & Biases
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    
    run = wandb.init(
        project=args.wandb_project,
        entity="ashetty21-university-of-california-berkeley",
        config={
            "model": "RFDETRSmall",
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": 1e-4,
            "grad_accum_steps": 4
        }
    )

    model = RFDETRSmall()

    # Dataset
    zip_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {args.dataset_dir}")
    dataset_zip = os.path.join(args.dataset_dir, zip_files[0])
    extract_dir = extract_dataset(dataset_zip)
    dataset_dir = restructure_dataset(extract_dir)

    # Train with wandb logging
    model.train(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=args.out_dir,
        wandb=True,
    )
    
    # Save final model as artifact
    print("Saving model artifact to W&B...")
    
    # Check if run is still active, if not resume it
    if wandb.run is None:
        api = wandb.Api()
        runs = api.runs(f"ashetty21-university-of-california-berkeley/{args.wandb_project}")
        latest_run = runs[0]
        
        run = wandb.init(
            project=args.wandb_project,
            entity="ashetty21-university-of-california-berkeley",
            id=latest_run.id,
            resume="allow"
        )
    
    artifact = wandb.Artifact(
        name=f"rfdetr-small-model",
        type="model",
        description=f"RF-DETR Small trained for {args.epochs} epochs"
    )
    
    # Add checkpoint files to artifact
    for file in os.listdir(args.out_dir):
        if file.endswith('_best_total.pth'):
            artifact.add_file(os.path.join(args.out_dir, file))
    
    # Log the artifact
    run.log_artifact(artifact)
    
    # Finish the run
    wandb.finish()
    print("✅ Model artifact saved to W&B")
  
  
def main():
    parser = argparse.ArgumentParser(description='Train RF-DETR on Flex AI')
    parser.add_argument('--dataset_dir', type=str, default='/input')
    parser.add_argument('--out_dir', type=str, 
                        default=os.environ.get('FLEXAI_OUTPUT_CHECKPOINT_DIR', '/output-checkpoints'),
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--wandb_project', type=str, default='rfdetr-training',
                        help='W&B project name')

    args = parser.parse_args()
    train_rfdetr(args)

if __name__ == '__main__':
    main()
