try:
    from .ikfastpy import PyKinematics
except ImportError:
    # Fallback for direct import
    from ikfastpy import PyKinematics

__all__ = ['PyKinematics']