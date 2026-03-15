import os


def get_sample_photos(sample_dir: str = "sample_photos") -> list[str]:
    """Return sorted absolute paths to image files in *sample_dir*.

    *sample_dir* is resolved relative to the project root (one level
    above this ``src/`` package).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, sample_dir)

    if not os.path.isdir(full_path):
        return []

    extensions = (".png", ".jpg", ".jpeg", ".webp")
    return sorted(
        os.path.join(full_path, f)
        for f in os.listdir(full_path)
        if f.lower().endswith(extensions)
    )
