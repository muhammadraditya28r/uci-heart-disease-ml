import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from collections.abc import Sequence, Callable



def create_plot(plot: Callable[..., Axes], 
                data: pd.DataFrame, 
                columns: Sequence[str], hue: str | None = None, 
                title: str | None = None,
                **kwargs) -> None:
    """
    Reusable function for creating multiple plot
    """

    rows = (len(columns) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))

    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        plot(data=data, x=col, ax=ax, hue=hue, **kwargs)
        plot_title = f"{title} {col.title()}" if title else col.title()
        ax.set_title(plot_title)

    for ax in axes[len(columns):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
