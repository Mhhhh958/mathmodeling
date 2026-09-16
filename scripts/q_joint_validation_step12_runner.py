import q_joint_validation_step12 as q

_original_load = q.joblib.load

def _load_pipeline(path):
    obj = _original_load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj

q.joblib.load = _load_pipeline

if __name__ == "__main__":
    q.main()
