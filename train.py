import os
import zipfile
import argparse
import yaml
from pathlib import Path
from rfdetr import RFDETRBase

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
    import json
    
    dataset_dir = os.path.join(extract_dir, 'dataset')
    
    # Rename val to valid
    if os.path.exists(os.path.join(dataset_dir, 'val')):
        os.rename(os.path.join(dataset_dir, 'val'), 
                  os.path.join(dataset_dir, 'valid'))
    
    # Move annotations into their respective folders
    for split in ['train', 'valid', 'test']:
        anno_file = os.path.join(dataset_dir, f'annotations_{split}.json')
        if split == 'valid':
            # Handle val/valid naming
            anno_file_alt = os.path.join(dataset_dir, 'annotations_val.json')
            if os.path.exists(anno_file_alt):
                anno_file = anno_file_alt
        
        if os.path.exists(anno_file):
            dest = os.path.join(dataset_dir, split, '_annotations.coco.json')
            shutil.move(anno_file, dest)
    
    return dataset_dir
  
def train_rfdetr(args):

    model = RFDETRBase()

    # Dataset
    zip_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {args.dataset_dir}")
    dataset_zip = os.path.join(args.dataset_dir, zip_files[0])
    extract_dir = extract_dataset(dataset_zip)
    dataset_dir = restructure_dataset(extract_dir)

    model.train(
        dataset_dir=dataset_dir,
        epochs=10,
        batch_size=16,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=args.out_dir
    )
  
  
def main():
    parser = argparse.ArgumentParser(description='Train RF-DETR on Flex AI')
    parser.add_argument('--dataset_dir', type=str, default='/input')
    parser.add_argument('--out_dir', type=str, 
                        default=os.environ.get('FLEXAI_OUTPUT_CHECKPOINT_DIR', '/output-checkpoints'),
                        help='Output directory for checkpoints')
    # parser.add_argument('--model', type=str, default='yolo11n.pt',
    #                     choices=['yolo11n.pt','yolo11s.pt','yolo11m.pt','yolo11l.pt','yolo11x.pt'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=16)

    args = parser.parse_args()
    train_rfdetr(args)

if __name__ == '__main__':
    main()
