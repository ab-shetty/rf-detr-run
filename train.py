import os
import zipfile
import argparse
import yaml
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

    model = RFDETRSmall()

    # Dataset
    zip_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {args.dataset_dir}")
    dataset_zip = os.path.join(args.dataset_dir, zip_files[0])
    extract_dir = extract_dataset(dataset_zip)
    dataset_dir = restructure_dataset(extract_dir)

    model.train(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch,
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)

    args = parser.parse_args()
    train_rfdetr(args)

if __name__ == '__main__':
    main()
