"""Initialise the Great Expectations file-backed data context."""
import great_expectations as gx

context = gx.get_context(mode="file", project_root_dir="great_expectations")
print("context root:", context.root_directory)
