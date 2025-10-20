import sys

SCRIPTS_PATH = '/kaggle/input/arc-prize-2025-arcsolver-4b-scripts/Code'


def subprocess_wrapper(args):
    if SCRIPTS_PATH not in sys.path:
        sys.path.append(SCRIPTS_PATH)

    import solver
    solver.solve_task_chunk(args)