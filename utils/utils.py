
import os
def extract_filename(path):
    """
    Extract the base name from a given path and remove the file extension and hyphens.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    file_name_no_hyphens : str
        The filename without extension and hyphens.
    """
    # Extract the base name from the path
    base_name = os.path.basename(path)
    # Remove the file extension
    file_name = os.path.splitext(base_name)[0]
    # Remove hyphens
    file_name_no_hyphens = file_name.replace("-", "")
    return file_name_no_hyphens