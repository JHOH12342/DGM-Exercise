import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.grid": False})

__all__ = ["matplotlib", "plt"]
