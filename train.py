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

def train_rfdetr(args):

    model = RFDETRBase()

    # Dataset
    zip_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {args.dataset_dir}")
    dataset_zip = os.path.join(args.dataset_dir, zip_files[0])
    extract_dir = extract_dataset(dataset_zip)

    model.train(
        dataset_dir=extract_dir,
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
