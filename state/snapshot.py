import os
from dotenv import load_dotenv
from storage.minio_client import get_minio_client

load_dotenv()

def generate_snapshot() -> dict:
    """
    List all objects in the configured MinIO bucket and return a snapshot
    of their metadata — without downloading any file content.
 
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
        snapshot[obj.object_name]={
            "etag":obj.etag,
            "last_modified":obj.last_modified.isoformat(),
        }

    return snapshot