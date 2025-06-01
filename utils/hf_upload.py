import fiftyone as fo
from fiftyone.utils.huggingface import push_to_hub
from dotenv import load_dotenv
import os

def upload_to_huggingface(
    dataset: fo.Dataset,
    repo_name: str,
    description: str = None,
    organization: str = None,
    hf_token: str = None,
    license: str = None,
    tags: list = None,
    private: bool = True,
    exist_ok: bool = False,
    dataset_type: str = None,
    min_fiftyone_version: str = None,
    label_field: str = None,
    frame_labels_field: str = None,
    preview_path: str = None,
    chunk_size: int = None,
    **data_card_kwargs
):
    """
    Upload a FiftyOne dataset to HuggingFace using the official FiftyOne integration

    Args:
        dataset (fo.Dataset): FiftyOne dataset to upload
        repo_name (str): Name for the repository
        description (str, optional): Description for the HuggingFace repository
        organization (str, optional): Organization name if uploading to an org
        hf_token (str, optional): HuggingFace API token
        license (str, optional): The license of the dataset
        tags (list, optional): A list of tags for the dataset
        private (bool, optional): Whether the repo should be private. Defaults to True
        exist_ok (bool, optional): If True, don't raise error if repo exists. Defaults to False
        dataset_type (str, optional): The type of dataset to create
        min_fiftyone_version (str, optional): Minimum FiftyOne version required (e.g. "0.23.0")
        label_field (str, optional): Controls which label field(s) to export
        frame_labels_field (str, optional): Controls which frame label field(s) to export
        preview_path (str, optional): Path to preview image/video for readme
        chunk_size (int, optional): Number of media files per subdirectory
        **data_card_kwargs: Additional keyword arguments for DatasetCard constructor
    """
    load_dotenv()
    
    # Get token from environment
    if not hf_token:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
    if not hf_token:
        raise ValueError(
            "No HuggingFace token found. Please provide it either through the "
            "hf_token parameter or set it as HUGGINGFACE_TOKEN environment variable"
        )
    
    # Construct repository ID
    repo_id = f"{organization}/{repo_name}" if organization else repo_name
    
    # Push dataset to HuggingFace
    push_to_hub(
        dataset,
        repo_id,
        description=description,
        license=license,
        tags=tags,
        private=private,
        exist_ok=exist_ok,
        dataset_type=dataset_type,
        min_fiftyone_version=min_fiftyone_version,
        label_field=label_field,
        frame_labels_field=label_field,
        access_token=hf_token,
        preview_path=preview_path,
        chunk_size=chunk_size,
        **data_card_kwargs
    ) 