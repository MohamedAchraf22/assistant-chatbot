def compare_states(old_state:dict,current_state:dict)->dict:
     """
    Compare two ingestion state snapshots and classify changes.
 
    Args:
        old_state:     Snapshot from the previous ingestion run.
        current_state: Snapshot from the current run.
 
    Returns:
        {
            "new_files":     [<object_name>, ...],  # present in current, absent in old
            "updated_files": [<object_name>, ...],  # present in both, but etag changed
            "deleted_files": [<object_name>, ...]   # present in old, absent in current
        }
    """
     old_keys=set(old_state.keys())
     current_keys=set(current_state.keys())

     new_files=sorted(current_keys-old_keys)
     deleted_files=sorted(old_keys-current_keys)
     updated_files=sorted(
          name
          for name in old_keys& current_keys
          if old_state[name]['etag']!=current_state[name]['etag']
     )

     return{
          "new_files":new_files,
          "updated_files":updated_files,
          "deleted_files":deleted_files,
     }