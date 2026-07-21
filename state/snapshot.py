import os
from dotenv import load_dotenv
from storage.minio_client import get_minio_client
from config import SUPPORTED_EXTENSIONS

load_dotenv()

def generate_snapshot() -> dict:
    """
    List all objects in the configured MinIO bucket and return a snapshot
    of their metadata — without downloading any file content.

    Objects whose extension is not in SUPPORTED_EXTENSIONS are excluded
    entirely: they never enter the snapshot, so compare_states() can never
    classify them as new/updated, sync() never attempts to load them, and
    they never get written to ingestion_state.json.

    Returns:
        {
            "<object_name>": {
                "etag": "<etag>",
                "last_modified": "<iso8601 timestamp>"
            },
            ...
        }
    """

    bucket_name=os.getenv('MINIO_BUCKET')
    client=get_minio_client()

    snapshot={}
    for obj in client.list_objects(bucket_name,recursive=True):
        _, ext = os.path.splitext(obj.object_name)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file:\n{obj.object_name}")
            continue

        snapshot[obj.object_name]={
            "etag":obj.etag,
            "last_modified":obj.last_modified.isoformat(),
        }

    return snapshot