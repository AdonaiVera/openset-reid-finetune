import fiftyone as fo
import random
from pathlib import Path
from utils.vlms import GeminiVLM
from utils.download_dataset import Market1501Downloader
from utils.hf_upload import upload_to_huggingface
from utils.tools import ImageDuplicateRemover
from fiftyone.types.dataset_types import FiftyOneDataset


def create_fiftyone_dataset(dataset_info, vlm):
    """Create a FiftyOne dataset with Gemini-generated descriptions"""
    try:
        # Init image duplicate remover
        remover = ImageDuplicateRemover() 

        # Create dataset
        dataset = fo.Dataset("fiftyone-reid-global-attributes")
        
        # Keep track of processed image paths
        processed_paths = set()
        
        # Group images by person_id for batch processing
        person_images = {}
        for subset, samples in dataset_info.items():
            for sample_info in samples:
                pid = sample_info['person_id']
                if pid not in person_images:
                    person_images[pid] = {'subset': subset, 'images': []}

                if sample_info['image_path'] not in processed_paths:
                    person_images[pid]['images'].append(sample_info)
                    processed_paths.add(sample_info['image_path'])
        
        # Process each person's images
        total_persons = len(person_images)
        for idx, (pid, person_data) in enumerate(person_images.items(), 1):
            try:
                print(f"Processing person {idx}/{total_persons} (ID: {pid})")
                
                # Get all image paths for this person and remove duplicates
                image_paths = [info['image_path'] for info in person_data['images']]
                unique_image_paths = remover.remove_repeated_images(image_paths)

                if len(unique_image_paths) > 5:
                    unique_image_paths = random.sample(unique_image_paths, 5)
                
                # Generate attributes and description for unique images of this person
                attributes, description = vlm.generate_global_attributes_batch(unique_image_paths)
                
                if description == "Unknown":
                    continue

                for sample_info in person_data['images']:
                    try:
                        if sample_info['image_path'] in unique_image_paths:
                            print(sample_info['image_path'])
                            sample = fo.Sample(filepath=sample_info['image_path'])
                            sample["tags"] = [person_data['subset']]
                            sample["person_id"] = sample_info['person_id']
                            sample["camera_id"] = sample_info['camera_id']
                            sample["attributes"] = attributes
                            sample["description"] = description
                            dataset.add_sample(sample)
                    except Exception as e:
                        print(f"Error processing sample {sample_info['image_path']}: {e}")
                        continue

            except Exception as e:
                print(f"Error processing person {pid}: {e}")
                continue
                
        return dataset
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None

if __name__ == "__main__":
    # Initialize the Gemini VLM
    vlm = GeminiVLM()
    
    # Get the prepared dataset info
    downloader = Market1501Downloader()
    dataset_info = downloader.download_and_prepare()
    
    # Create FiftyOne dataset with descriptions
    dataset = create_fiftyone_dataset(dataset_info, vlm)

    print(dataset.get_field_schema())

    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(dataset.match_tags('train'))}")
    print(f"Query samples: {len(dataset.match_tags('query'))}")
    print(f"Gallery samples: {len(dataset.match_tags('gallery'))}")
    
    # Launch the FiftyOne app to visualize the dataset
    session = fo.launch_app(dataset) 

    input("When you finished to visualize the dataset, press Enter or any key to continue...")

    print("Closing the FiftyOne app...")

    description = """
        Market-1501 Person Re-identification Dataset with Gemini-generated Attributes
        
        This dataset contains person re-identification images from the Market-1501 dataset,
        enhanced with detailed attribute descriptions generated using Google's Gemini Vision model.
        Each person has associated attributes and natural language descriptions that capture
        their visual characteristics.
    """
    
    upload_to_huggingface(
        dataset=dataset,
        repo_name="fiftyone-multiview-reid-attributes",
        description=description,
        tags=[
            "person-reid",
            "market1501",
            "attributes",
            "gemini",
            "computer-vision",
            "person-description",
            "multiview",
            "open-set"
        ],
        private=False,
        exist_ok=True,
        dataset_type=FiftyOneDataset,
        min_fiftyone_version="1.5.2",
        label_field=["attributes", "description", "person_id", "camera_id", "tags"],
        preview_path=None
    )
    
    session.close()