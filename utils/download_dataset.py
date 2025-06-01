from pathlib import Path
import hashlib
import shutil
import re
from glob import glob
import os.path as osp
from zipfile import ZipFile

class Market1501Downloader:
    URL = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
    MD5 = '65005ab7d12ec1c44de4eeafe813e68a'
    
    def __init__(self, root_dir="dataset"):
        self.root = Path(root_dir)
        self.raw_dir = self.root / 'raw'
        self.images_dir = self.root / 'images'
    
    def is_dataset_prepared(self):
        """Check if dataset is already downloaded and prepared"""
        if not self.images_dir.exists():
            return False
        image_files = list(self.images_dir.glob('*.jpg'))
        return len(image_files) > 0
    
    def download_and_prepare(self):
        """Download and prepare the Market1501 dataset"""        
        self._create_directories()
        zip_path = self._get_dataset_file()
        extract_dir = self._extract_dataset(zip_path)
        return self._organize_dataset(extract_dir)
    
    def _create_directories(self):
        """Create necessary directories"""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_dataset_file(self):
        """Get the dataset zip file (manual download required)"""
        zip_path = self.raw_dir / 'Market-1501-v15.09.15.zip'
        
        if zip_path.exists():
            file_md5 = hashlib.md5(open(zip_path, 'rb').read()).hexdigest()
            if file_md5 == self.MD5:
                print("Using existing downloaded file")
                return zip_path
        
        raise RuntimeError(
            f"Please download the dataset manually from {self.URL} "
            f"and place it at {zip_path}"
        )
    
    def _extract_dataset(self, zip_path):
        """Extract the dataset zip file"""
        extract_dir = self.raw_dir / 'Market-1501-v15.09.15'
        
        if not extract_dir.exists():
            print("Extracting zip file...")
            with ZipFile(zip_path) as z:
                z.extractall(path=self.raw_dir)
        
        return extract_dir
    
    def _organize_dataset(self, extract_dir):
        """Organize dataset into a structured format"""
        pattern = re.compile(r'([-\d]+)_c(\d)')
        dataset_info = {
            'train': [],
            'query': [],
            'gallery': []
        }
        
        # Process each subset
        for subset in ['bounding_box_train', 'query', 'bounding_box_test']:
            target_subset = 'gallery' if subset == 'bounding_box_test' else \
                          'train' if subset == 'bounding_box_train' else 'query'
            
            fpaths = sorted(glob(str(extract_dir / subset / '*.jpg')))
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                
                # Skip junk images (pid == -1)
                if pid == -1:
                    continue
                
                # Copy image to organized directory
                new_fname = f"{pid:08d}_{cam:02d}.jpg"
                new_path = self.images_dir / new_fname
                shutil.copy(fpath, new_path)
                
                # Store image info
                dataset_info[target_subset].append({
                    'image_path': str(new_path),
                    'person_id': pid,
                    'camera_id': cam
                })
        
        return dataset_info
    
    def _get_dataset_info(self):
        """Get dataset info for already prepared dataset"""
        pattern = re.compile(r'([-\d]+)_c(\d)')
        dataset_info = {
            'train': [],
            'query': [],
            'gallery': []
        }
        
        # Process all images in the organized directory
        for image_path in sorted(self.images_dir.glob('*.jpg')):
            fname = image_path.name
            pid, cam = map(int, pattern.search(fname).groups())
            
            # Skip junk images (pid == -1)
            if pid == -1:
                continue
            
            # Determine subset based on person_id range
            if 0 <= pid <= 750: 
                subset = 'train'
            elif 751 <= pid <= 1501:  
                subset = 'gallery'  
                
                # Query images are typically one image per person per camera
                if len(list(self.images_dir.glob(f'{pid:08d}_*.jpg'))) == 1:
                    subset = 'query'
            
            dataset_info[subset].append({
                'image_path': str(image_path),
                'person_id': pid,
                'camera_id': cam
            })
        
        return dataset_info