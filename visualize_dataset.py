import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

def visualize_dataset():
    """
    Load and visualize the dataset from HuggingFace hub
    """
    print("Loading dataset from HuggingFace hub...")
    
    # Load the dataset from HuggingFace
    dataset = load_from_hub(
        repo_id="adonaivera/fiftyone-multiview-reid-attributes",
        dataset_name="fiftyone-multiview-reid2",
        overwrite=True
    )
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(dataset.match_tags('train'))}")
    print(f"Query samples: {len(dataset.match_tags('query'))}")
    print(f"Gallery samples: {len(dataset.match_tags('gallery'))}")
    
    # Launch the FiftyOne app
    session = fo.launch_app(dataset)
    
    input("\nFiftyOne app launched. Press Enter to close the app...")
    
    session.close()

if __name__ == "__main__":
    visualize_dataset()
