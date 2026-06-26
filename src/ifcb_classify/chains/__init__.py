"""Optional YOLO-based chain counting for IFCB chain-forming taxa.

This subpackage is only needed when counting individual cells in chain-forming
plankton (e.g. Skeletonema). It requires the optional ``chains`` extra:

    uv pip install -e ".[chains]"

Nothing here imports ``ultralytics`` at module load time, so the core
classifier never pays for it unless chain counting is actually invoked.
"""
