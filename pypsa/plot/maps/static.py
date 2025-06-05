"""Map plots for network objects."""

from __future__ import annotations

import logging
import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, FancyArrow, Patch, Polygon, Wedge

from pypsa.common import _convert_to_series, deprecated_kwargs
from pypsa.constants import DEFAULT_EPSG
from pypsa.geo import (
    compute_bbox,
    get_projected_area_factor,
)
from pypsa.plot.maps.common import apply_layouter

cartopy_present = True
try:
    import cartopy
    import cartopy.mpl.geoaxes
    from cartopy.mpl.geoaxes import GeoAxesSubplot
except ImportError:
    cartopy_present = False
    GeoAxesSubplot = Any


if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx
    from matplotlib.legend import Legend

    from pypsa.components.components import Components
    from pypsa.networks import Network

logger = logging.getLogger(__name__)


def _apply_cmap(
    colors: pd.Series,
    cmap: str | mcolors.Colormap | None,
    cmap_norm: mcolors.Normalize | None = None,
) -> pd.Series:
    if np.issubdtype(colors.dtype, np.number):
        if not isinstance(cmap, mcolors.Colormap):
            cmap = plt.get_cmap(cmap)
        if not cmap_norm:
            cmap_norm = plt.Normalize(vmin=colors.min(), vmax=colors.max())
        colors = colors.apply(lambda cval: cmap(cmap_norm(cval)))
    return colors


class MapPlotter:
    """Class to plot a PyPSA network on a map."""

    def __init__(
        self,
        n: Network,
        layout: Callable[[nx.Graph], dict[Any, tuple[float, float]]] | None = None,
        boundaries: tuple[float, float, float, float] | None = None,
        margin: float = 0.05,
        jitter: float | None = None,
        buses: pd.Index | None = None,
    ) -> None:
        """Initialize MapPlotGenerator instance.

        Parameters
        ----------
        n : Network
            The PyPSA network to plot
        layout : networkx.drawing.layout, optional
            Layout function to use for node positioning
        boundaries : tuple[float, float, float, float], optional
            Plot boundaries as [xmin, xmax, ymin, ymax]
        margin : float, default 0.05
            Margin around plot boundaries
        jitter : float, optional
            Amount of random noise to add to node positions
        buses : pd.Index, optional
            Subset of buses to plot

        """
        """Initialize MapPlotGenerator instance.

        Parameters
        ----------
        n : Network
            The PyPSA network to plot
        layout : networkx.drawing.layout, optional
            Layout function to use for node positioning
        boundaries : tuple[float, float, float, float], optional
            Plot boundaries as [xmin, xmax, ymin, ymax]
        margin : float, default 0.05
            Margin around plot boundaries
        jitter : float, optional
            Amount of random noise to add to node positions
        buses : pd.Index, optional
            Subset of buses to plot
        """
        self._n = n
        self._x = None
        self._y = None
        self._layout = layout
        self._boundaries = boundaries
        self._margin = margin
        self._ax: Axes | GeoAxesSubplot | None = None
        self._area_factor = 1.0

        if jitter:
            self.add_jitter(jitter)

    @property
    def n(self) -> Network:
        """Get the network object."""
        return self._n

    @property
    def x(self) -> pd.Series:
        """Get the x-coordinates of the buses."""
        if self._x is None:
            self.set_layout()
        return self._x

    @x.setter
    def x(self, value: pd.Series) -> None:
        """Set the x-coordinates of the buses."""
        self._x = value

    @property
    def y(self) -> pd.Series:
        """Get the y-coordinates of the buses."""
        if self._y is None:
            self.set_layout()
        return self._y

    @y.setter
    def y(self, value: pd.Series) -> None:
        """Set the y-coordinates of the buses."""
        self._y = value

    @property
    def margin(self) -> float:
        """Get the margin around the plot boundaries."""
        return self._margin

    @margin.setter
    def margin(self, value: float) -> None:
        """Set the margin around the plot boundaries."""
        self._margin = value

    @property
    def boundaries(self) -> tuple[float, float, float, float] | None:
        """Get the plot boundaries."""
        if self._boundaries is None:
            self.set_boundaries(self._boundaries, self.margin, self._n.buses.index)
        return self._boundaries

    @boundaries.setter
    def boundaries(
        self,
        value: tuple[float, float, float, float] | None,
    ) -> None:
        """Set the plot boundaries."""
        if value is not None and len(value) != 4:
            msg = "Boundaries must be a sequence of 4 values (xmin, xmax, ymin, ymax)"
            raise ValueError(msg)
        self._boundaries = value

    @property
    def ax(
        self,
    ) -> Axes | GeoAxesSubplot | None:
        """Get the axis for plotting."""
        return self._ax

    @ax.setter
    def ax(
        self,
        value: Axes | GeoAxesSubplot | None,
    ) -> None:
        """Set the axis for plotting."""
        if not cartopy_present:
            axis_type = (Axes,)
        else:
            axis_type = (Axes, GeoAxesSubplot)  # type: ignore
        if value is not None and not isinstance(value, axis_type):
            msg = "ax must be either matplotlib Axes or GeoAxesSubplot"
            raise ValueError(msg)
        self._ax = value

    @property
    def area_factor(self) -> float:
        """Get the area factor for scaling."""
        return self._area_factor

    @area_factor.setter
    def area_factor(self, value: float | None) -> None:
        """Set the area factor for scaling."""
        if value is not None and not isinstance(value, int | float):
            msg = "area_factor must be a number"
            raise ValueError(msg)
        self._area_factor = float(value) if value is not None else 1.0

    def set_layout(self, layouter: Callable | None = None) -> None:
        """Set the layout for node positions.

        Parameters
        ----------
        layouter : networkx.drawing.layout, optional
            Layout function to use. If None, uses planar layout if possible,
            otherwise Kamada-Kawai layout.

        """
        """Set the layout for node positions.

        Parameters
        ----------
        layouter : networkx.drawing.layout, optional
            Layout function to use. If None, uses planar layout if possible,
            otherwise Kamada-Kawai layout.
        """
        # Check if networkx layouter is given or needed to get bus positions
        is_empty = (
            (self.n.buses[["x", "y"]].isnull() | (self.n.buses[["x", "y"]] == 0))
            .all()
            .all()
        )
        if layouter or self._layout or is_empty:
            self.x, self.y = apply_layouter(self.n, layouter, inplace=False)
        else:
            self.x, self.y = self.n.buses["x"], self.n.buses["y"]
        self.crs = self.n.crs

    def set_boundaries(
        self,
        boundaries: tuple[float, float, float, float] | None = None,
        margin: float = 0.05,
        buses: pd.Index | None = None,
    ) -> None:
        """Set the plot boundaries.

        Parameters
        ----------
        boundaries : tuple[float, float, float, float], optional
            Plot boundaries as [xmin, xmax, ymin, ymax]. If None, computed from bus positions.
        margin : float, default 0.05
            Margin around plot boundaries
        buses : pd.Index, optional
            Subset of buses to use for computing boundaries

        """
        """Set the plot boundaries.

        Parameters
        ----------
        boundaries : tuple[float, float, float, float], optional
            Plot boundaries as [xmin, xmax, ymin, ymax]. If None, computed from bus positions.
        margin : float, default 0.05
            Margin around plot boundaries
        buses : pd.Index, optional
            Subset of buses to use for computing boundaries
        """
        # Set boundaries, if not given

        if buses is None:
            buses = self.n.buses.index

        if boundaries is None:
            (x1, y1), (x2, y2) = compute_bbox(self.x[buses], self.y[buses], margin)
            self.boundaries = (x1, x2, y1, y2)
        else:
            self.boundaries = boundaries

    def init_axis(
        self,
        ax: Axes | None = None,
        projection: Any = None,
        geomap: bool = True,
        geomap_resolution: Literal["10m", "50m", "110m"] = "50m",
        geomap_colors: dict | bool | None = None,
        boundaries: tuple[float, float, float, float] | None = None,
        title: str = "",
    ) -> None:
        """Initialize the plot axis with geographic features if requested.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axis to plot on. If None, creates new axis.
        projection : cartopy.crs.Projection, optional
            Map projection to use
        geomap : bool, default True
            Whether to add geographic features
        geomap_resolution : {'10m', '50m', '110m'}, default '50m'
            Resolution of geographic features
        geomap_colors : dict or bool, optional
            Colors for geographic features
        boundaries : tuple[float, float, float, float], optional
            Plot boundaries as [xmin, xmax, ymin, ymax]
        title : str, default ""
            Plot title

        """
        # Ensure that boundaries are set
        if boundaries is None:
            boundaries = self.boundaries

        # Check if geomap is requested but cartopy not available
        if geomap and not cartopy_present:
            logger.warning(
                "Cartopy is not available. Falling back to non-geographic plotting."
            )
            geomap = False

        # Set up plot (either cartopy or matplotlib)
        if geomap:
            network_projection = cartopy.crs.Projection(self.n.crs)
            if projection is None:
                projection = network_projection
            elif not isinstance(projection, cartopy.crs.Projection):
                msg = "The passed projection is not a cartopy.crs.Projection"
                raise ValueError(msg)

            if ax is None:
                self.ax = plt.axes(projection=projection)
            elif not isinstance(ax, GeoAxesSubplot):
                msg = "The passed axis is not a GeoAxesSubplot. You can "
                "create one with: \nimport cartopy.crs as ccrs \n"
                "fig, ax = plt.subplots("
                'subplot_kw={"projection":ccrs.PlateCarree()})'
                raise ValueError(msg)
            else:
                self.ax = ax

            # Transform bus positions to projection, track the new crs
            x, y, _ = self.ax.projection.transform_points(  # type: ignore
                network_projection, self.x.values, self.y.values
            ).T
            self.x_trans, self.y_trans = (
                pd.Series(x, self.x.index),
                pd.Series(y, self.y.index),
            )

            if geomap_colors is not False:
                if geomap_colors is None:
                    geomap_colors = {}
                if isinstance(geomap_colors, dict):
                    self.add_geomap_features(geomap_resolution, geomap_colors)
                else:
                    self.add_geomap_features(geomap_resolution)

            self.ax.set_extent(boundaries, crs=network_projection)  # type: ignore

            self.area_factor = get_projected_area_factor(self.ax, self.n.srid)
        else:
            if ax is None:
                self.ax = plt.gca()
            else:
                self.ax = ax
            self.ax.axis(boundaries)
            self.x_trans, self.y_trans = self.x, self.y
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title(title)

    def add_geomap_features(
        self,
        resolution: Literal["10m", "50m", "110m"] = "50m",
        geomap_colors: dict | None = None,
    ) -> None:
        """Add geographic features to the map using cartopy.

        Parameters
        ----------
        resolution : {'10m', '50m', '110m'}, default '50m'
            Resolution of geographic features
        geomap_colors : dict, optional
            Colors for geographic features. Keys can include:
            - 'ocean': color for ocean areas
            - 'land': color for land areas
            - 'border': color for country borders
            - 'coastline': color for coastlines

        """
        """Add geographic features to the map using cartopy."""
        if not cartopy_present:
            logger.warning("Cartopy is not available. Cannot add geographic features.")
            return

        if not isinstance(self.ax, GeoAxesSubplot):
            msg = "The axis must be a GeoAxesSubplot to add geographic features."
            raise TypeError(msg)

        if resolution not in ["10m", "50m", "110m"]:
            msg = "Resolution has to be one of '10m', '50m', '110m'"
            raise ValueError(msg)

        if geomap_colors is None:
            geomap_colors = {
                "ocean": "lightblue",
                "land": "whitesmoke",
                "border": "darkgray",
                "coastline": "black",
            }

        if "land" in geomap_colors:
            self.ax.add_feature(
                cartopy.feature.LAND.with_scale(resolution),
                facecolor=geomap_colors["land"],
            )

        if "ocean" in geomap_colors:
            self.ax.add_feature(
                cartopy.feature.OCEAN.with_scale(resolution),
                facecolor=geomap_colors["ocean"],
            )

        self.ax.add_feature(
            cartopy.feature.BORDERS.with_scale(resolution),
            linewidth=0.3,
            edgecolor=geomap_colors.get("border", "black"),
        )

        self.ax.add_feature(
            cartopy.feature.COASTLINE.with_scale(resolution),
            linewidth=0.3,
            edgecolor=geomap_colors.get("coastline", "black"),
        )

    def add_jitter(self, jitter: float) -> tuple[pd.Series, pd.Series]:
        """Add random jitter to data.

        Parameters
        ----------
        jitter : float
            The amount of jitter to add. Function adds a random number between -jitter and
            jitter to each element in the data arrays.

        Returns
        -------
        x_jittered : numpy.ndarray
            X data with added jitter.
        y_jittered : numpy.ndarray
            Y data with added jitter.

        """
        rng = np.random.default_rng()  # Create a random number generator
        self.x = self.x + rng.uniform(low=-jitter, high=jitter, size=len(self.x))
        self.y = self.y + rng.uniform(low=-jitter, high=jitter, size=len(self.y))

        return self.x, self.y

    def get_multiindex_buses(
        self,
        sizes: pd.Series,
        colors: pd.Series,
        alpha: float | pd.Series,
        split_circles: bool,
    ) -> list[Wedge]:
        """Create patches for buses with multi-indexed size data.

        Parameters
        ----------
        sizes : pd.Series
            Series with multi-index where first level represents buses
            and second level represents components/carriers.
        colors : pd.Series
            Series with colors for each component/carrier.
        alpha : float | pd.Series
            Transparency value(s) for the patches.
        split_circles : bool
            Whether to split the circles into halves for positive/negative values.

        Returns
        -------
        list[Wedge]
            List of Wedge patches representing the buses.

        """
        # We are drawing pies to show all the different shares
        patches = []
        for b_i in sizes.index.unique(level=0):
            s_base = sizes.loc[b_i]

            if split_circles:
                # As only half of the circle is drawn, increase area by factor 2
                s_base = s_base * 2
                s_base = (
                    s_base[s_base > 0],
                    s_base[s_base < 0],
                )
                starts: tuple[float, ...] = 0.0, 1.0
                scope = 180
            else:
                s_base = (s_base[s_base > 0],)
                starts = (0.25,)
                scope = 360

            for s, start in zip(s_base, starts, strict=False):
                radius = abs(s.sum()) ** 0.5
                ratios = abs(s) if radius == 0.0 else s / s.sum()
                for i, ratio in ratios.items():
                    patches.append(
                        Wedge(
                            (self.x_trans.at[b_i], self.y_trans.at[b_i]),
                            radius,
                            scope * start,
                            scope * (start + ratio),
                            facecolor=colors[i],
                            alpha=alpha,
                        )
                    )
                    start = start + ratio
        return patches

    def get_singleindex_buses(
        self,
        sizes: pd.Series,
        colors: pd.Series,
        alpha: float | pd.Series,
    ) -> list[Circle]:
        """Create patches for buses with single-indexed size data.

        Parameters
        ----------
        sizes : pd.Series
            Series with bus indices representing sizes of buses.
        colors : pd.Series
            Series with colors for each bus.
        alpha : float | pd.Series
            Transparency value(s) for the patches.

        Returns
        -------
        list[Circle]
            List of Circle patches representing the buses.

        """
        patches = []
        for b_i in sizes.index[(sizes != 0) & ~sizes.isna()]:
            radius = sizes.at[b_i] ** 0.5
            patches.append(
                Circle(
                    (self.x_trans.at[b_i], self.y_trans.at[b_i]),
                    radius,
                    facecolor=colors.at[b_i],
                    alpha=alpha,
                )
            )
        return patches

    @staticmethod
    def _dataframe_from_arguments(index: pd.Index, **kwargs: Any) -> pd.DataFrame:
        if any(isinstance(v, pd.Series) for v in kwargs.values()):
            return pd.DataFrame(kwargs)
        return pd.DataFrame(kwargs, index=index)

    def _flow_ds_from_arg(
        self,
        flow: pd.Series | str | float | Callable | None,
        c_name: str,
    ) -> pd.Series | None:
        """Convert flow argument to pandas Series.

        Parameters
        ----------
        flow : Series|str|int|float|callable|None
            Flow data to convert
        c_name : str
            Name of the network component

        Returns
        -------
        pd.Series | None
            Converted flow data as Series, or None if flow was None

        """
        if isinstance(flow, pd.Series):
            return flow

        if flow in self.n.snapshots:
            return self.n.dynamic(c_name).p0.loc[flow]

        if isinstance(flow, str) or callable(flow):
            return self.n.dynamic(c_name).p0.agg(flow, axis=0)

        if isinstance(flow, int | float):
            return pd.Series(flow, index=self.n.static(c_name).index)

        if flow is not None:
            msg = f"The 'flow' argument must be a pandas.Series, a string, a float or a callable, got {type(flow)}."
            raise ValueError(msg)

        return None

    def get_branch_collection(
        self,
        c: Components,
        widths: float | pd.Series,
        colors: str | pd.Series,
        alpha: float | pd.Series,
        geometry: pd.Series,
        auto_scale_branches: bool = True,
    ) -> LineCollection | PatchCollection:
        """Create a collection of branches for a single component.

        Parameters
        ----------
        c : Component
            Network component being plotted
        widths : float/Series
            Line widths for the component
        colors : str/Series
            Colors for the component
        alpha : float/Series
            Alpha values for the component
        geometry : bool
            Whether to use geometry data for plotting
        auto_scale_branches : bool
            Whether to create a LineCollection with widths in data units
            or a PatchCollection with widths in display units

        """
        if auto_scale_branches:
            return self._get_branch_collection_lines(c, widths, colors, alpha, geometry)
        if geometry:
            msg = "The 'geometry' argument cannot be used with 'auto_scale_branches=True'."
            raise ValueError(msg)
        return self._get_branch_collection_patches(c, widths, colors, alpha)

    def _get_branch_collection_lines(
        self,
        c: Components,
        widths: pd.Series,
        colors: pd.Series,
        alpha: pd.Series,
        geometry: pd.Series,
    ) -> LineCollection:
        """Create a LineCollection for a single branch component."""
        idx = widths.index
        if not geometry:
            segments: np.ndarray = np.asarray(
                (
                    (
                        c.static.bus0[idx].map(self.x_trans),
                        c.static.bus0[idx].map(self.y_trans),
                    ),
                    (
                        c.static.bus1[idx].map(self.x_trans),
                        c.static.bus1[idx].map(self.y_trans),
                    ),
                )
            ).transpose(2, 0, 1)
        else:
            from shapely.geometry import LineString
            from shapely.wkt import loads

            linestrings = geometry[lambda ds: ds != ""].map(loads)
            if not all(isinstance(ls, LineString) for ls in linestrings):
                msg = "The WKT-encoded geometry in the 'geometry' column must be "
                "composed of LineStrings"
                raise ValueError(msg)
            segments = np.asarray(list(linestrings.map(np.asarray)))

        return LineCollection(
            list(segments),
            linewidths=widths,
            antialiaseds=(1,),
            colors=colors,
            alpha=alpha,
        )

    def _get_branch_collection_patches(
        self,
        c: Components,
        widths: pd.Series,
        colors: pd.Series,
        alpha: pd.Series,
    ) -> PatchCollection:
        """Create a PatchCollection of polygons representing lines with widths."""
        idx = widths.index
        patches = []
        facecolors = []
        alphas = []
        x0s, y0s = c.static.bus0.map(self.x_trans), c.static.bus0.map(self.y_trans)
        x1s, y1s = c.static.bus1.map(self.x_trans), c.static.bus1.map(self.y_trans)

        for i in idx:
            x0, y0 = x0s[i], y0s[i]
            x1, y1 = x1s[i], y1s[i]
            width = widths[i] * self.area_factor

            # Calculate the direction vector
            dx = x1 - x0
            dy = y1 - y0
            length = np.hypot(dx, dy)
            if length == 0:
                continue  # Skip zero-length lines
            udx = dx / length
            udy = dy / length

            # Perpendicular vector scaled by half the width
            px = -udy * width / 2.0
            py = udx * width / 2.0

            # Define the corners of the rectangle
            corners = [
                (x0 + px, y0 + py),
                (x1 + px, y1 + py),
                (x1 - px, y1 - py),
                (x0 - px, y0 - py),
            ]

            polygon = Polygon(corners, closed=True)
            patches.append(polygon)
            facecolors.append(colors[i])
            alphas.append(alpha[i])

        patch_collection = PatchCollection(
            patches,
            facecolors=facecolors,
            edgecolors="none",
            alpha=alphas,
        )

        return patch_collection

    @staticmethod
    def _directed_flow(
        coords: pd.DataFrame,
        flow: pd.Series,
        color: pd.Series,
        area_factor: float,
        alpha: float = 1,
    ) -> PatchCollection:
        """Generate arrows from flow data."""
        # this function is used for diplaying arrows representing the network flow
        data = pd.DataFrame(
            {
                "arrowsize": flow.abs().pipe(np.sqrt).clip(lower=1e-8),
                "direction": np.sign(flow),
                "linelength": (
                    np.sqrt(
                        (coords.x1 - coords.x2) ** 2.0 + (coords.y1 - coords.y2) ** 2
                    )
                ),
            }
        )
        data = data.join(coords)
        data["arrowsize"] = data["arrowsize"].mul(area_factor)
        data["arrowtolarge"] = 1.5 * data.arrowsize > data.linelength
        # swap coords for negativ directions
        data.loc[data.direction == -1.0, ["x1", "x2", "y1", "y2"]] = data.loc[
            data.direction == -1.0, ["x2", "x1", "y2", "y1"]
        ].values
        if ((data.linelength > 0.0) & (~data.arrowtolarge)).any():
            data["arrows"] = data[(data.linelength > 0.0) & (~data.arrowtolarge)].apply(
                lambda ds: FancyArrow(
                    ds.x1,
                    ds.y1,
                    0.6 * (ds.x2 - ds.x1)
                    - ds.arrowsize * 0.75 * (ds.x2 - ds.x1) / ds.linelength,
                    0.6 * (ds.y2 - ds.y1)
                    - ds.arrowsize * 0.75 * (ds.y2 - ds.y1) / ds.linelength,
                    head_width=ds.arrowsize,
                ),
                axis=1,
            )
        data.loc[(data.linelength > 0.0) & (data.arrowtolarge), "arrows"] = data[
            (data.linelength > 0.0) & (data.arrowtolarge)
        ].apply(
            lambda ds: FancyArrow(
                ds.x1,
                ds.y1,
                0.001 * (ds.x2 - ds.x1),
                0.001 * (ds.y2 - ds.y1),
                head_width=ds.arrowsize,
            ),
            axis=1,
        )
        data = data.dropna(subset=["arrows"])
        return PatchCollection(
            data.arrows,
            color=color,
            alpha=alpha,
            edgecolors="black",
            linewidths=0,
            zorder=4,
        )

    def get_flow_collection(
        self,
        c: Components,
        flow: pd.Series,
        widths: pd.Series,
        colors: pd.Series,
        alpha: pd.Series,
    ) -> PatchCollection:
        """Create a flow arrow collection for a single branch component.

        Parameters
        ----------
        c : Component
            Network component being plotted
        flow : pd.Series
            Flow values for the component
        widths : float/Series
            Line widths to scale the arrows
        colors : str/Series
            Colors for the arrows
        alpha : float/Series
            Alpha values for the arrows

        """
        idx = widths.index
        coords = pd.DataFrame(
            {
                "x1": c.static.bus0[idx].map(self.x_trans),
                "y1": c.static.bus0[idx].map(self.y_trans),
                "x2": c.static.bus1[idx].map(self.x_trans),
                "y2": c.static.bus1[idx].map(self.y_trans),
            }
        )

        return self._directed_flow(coords, flow, colors, self.area_factor, alpha)

    @staticmethod
    def scaling_factor_from_area_contribution(
        area_contributions: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        target_area_fraction: float = 0.1,
    ) -> float:
        """Scale series for plotting.

        Makes sure that the total area of all area contributions
        takes up approximately the specified fraction of the plot area.

        Parameters
        ----------
        area_contributions : pd.Series
            Series containing the balance values for each bus
        x_min, x_max : float
            The x-axis extent of the plot
        y_min, y_max : float
            The y-axis extent of the plot
        target_area_fraction : float, optional
            Desired fraction of plot area to be covered by all circles (default: 0.3)

        Returns
        -------
        pd.Series
            Scaled values

        """
        plot_area = (x_max - x_min) * (y_max - y_min)
        target_total_circle_area = plot_area * target_area_fraction
        current_total_area = np.sum(np.abs(area_contributions))
        if current_total_area == 0:
            return 1
        return target_total_circle_area / current_total_area

    @staticmethod
    def aggregate_flow_by_connection(
        flow: pd.Series, branches: pd.DataFrame
    ) -> pd.Series:
        """Aggregate flow values by bus connections irrespective of direction.

        This method aggregates flow values from different branch components between
        the same pair of buses, ensuring consistent directional representation.

        Parameters
        ----------
        flow : pd.Series
            Series containing flow values indexed by branch components
        branches : pd.DataFrame
            DataFrame with bus0 and bus1 columns defining connections

        Returns
        -------
        pd.Series
            Aggregated flow values with consistent direction

        """
        if flow.empty:
            return flow
        connected_buses = branches.loc[flow.index, ["bus0", "bus1"]]
        sign_correction = np.where(connected_buses.bus0 < connected_buses.bus1, 1, -1)

        flow_sorted = flow * sign_correction
        buses_sorted = connected_buses.apply(sorted, axis=1).str.join(" - ")
        flow_grouped = (
            flow_sorted.groupby(buses_sorted).transform("sum") * sign_correction
        )
        flow_grouped = flow_grouped[buses_sorted.drop_duplicates().index]

        return flow_grouped

    @staticmethod
    def flow_to_width(flow: pd.Series, width_factor: float = 0.2) -> pd.Series:
        """Calculate the width of a line based on the flow value.

        Parameters
        ----------
        flow : float or pd.Series
            Flow values
        width_factor : float
            Ratio between the flow width and and line width (default: 0.2)

        """
        return abs(flow) ** 0.5 * width_factor

    def draw_map(  # noqa: D417
        self,
        ax: Axes | None = None,
        projection: Any = None,
        geomap: bool = True,
        geomap_resolution: Literal["10m", "50m", "110m"] = "50m",
        geomap_colors: dict | bool | None = None,
        title: str = "",
        boundaries: tuple[float, float, float, float] | None = None,
        branch_components: list | set | None = None,
        bus_sizes: float | dict | pd.Series = 2e-2,
        bus_split_circles: bool = False,
        bus_colors: str | dict | pd.Series = None,
        bus_cmap: str | mcolors.Colormap | None = None,
        bus_cmap_norm: mcolors.Normalize | None = None,
        bus_alpha: float | dict | pd.Series = 1,
        geometry: bool = False,
        line_flow: float | str | Callable | dict | pd.Series = None,
        line_colors: str | dict | pd.Series = "rosybrown",
        line_cmap: str | mcolors.Colormap = "viridis",
        line_cmap_norm: mcolors.Normalize | None = None,
        line_alpha: float | dict | pd.Series = 1,
        line_widths: float | dict | pd.Series = 1.5,
        link_flow: float | str | Callable | dict | pd.Series = None,
        link_colors: str | dict | pd.Series = "darkseagreen",
        link_cmap: str | mcolors.Colormap = "viridis",
        link_cmap_norm: mcolors.Normalize | None = None,
        link_alpha: float | dict | pd.Series = 1,
        link_widths: float | dict | pd.Series = 1.5,
        transformer_flow: float | str | Callable | dict | pd.Series = None,
        transformer_colors: str | dict | pd.Series = "orange",
        transformer_cmap: str | mcolors.Colormap = "viridis",
        transformer_cmap_norm: mcolors.Normalize | None = None,
        transformer_alpha: float | dict | pd.Series = 1,
        transformer_widths: float | dict | pd.Series = 1.5,
        flow: str | Callable | dict | pd.Series = None,
        auto_scale_branches: bool = True,
    ) -> dict:
        """Plot the network buses and lines using matplotlib and cartopy.

        Parameters
        ----------
        boundaries : list/tuple, default None
            Boundaries of the plot in format [x1, x2, y1, y2]
        ax : matplotlib.pyplot.Axes, defaults to None
            Axis to plot on. Defaults to plt.gca() if geomap is False, otherwise
            to plt.axes(projection=projection).
        geomap: bool/str, default True
            Switch to use Cartopy and draw geographical features.
            If string is passed, it will be used as a resolution argument,
            valid options are '10m', '50m' and '110m'.
        geomap_resolution : str, default '50m'
            Resolution of the geomap, only valid if cartopy is installed.
            Valid options are '10m', '50m' and '110m'.
        projection: cartopy.crs.Projection, defaults to None
            Define the projection of your geomap, only valid if cartopy is
            installed. If None (default) is passed the projection for cartopy
            is set to cartopy.crs.PlateCarree
        geomap_colors : dict/bool, default None
            Specify colors to paint land and sea areas in.
            If True, it defaults to `{'ocean': 'lightblue', 'land': 'whitesmoke'}`.
            If no dictionary is provided, colors are white.
            If False, no geographical features are plotted.
        title : string, default ""
            Graph title
        branch_components : list, default n.branch_components
            Branch components to be plotted
        bus_sizes : float/dict/pandas.Series
            Sizes of bus points, defaults to 1e-2. If a multiindexed Series is passed,
            the function will draw pies for each bus (first index level) with
            segments of different color (second index level). Such a Series is ob-
            tained by e.g. n.generators.groupby(['bus', 'carrier']).p_nom.sum()
        bus_split_circles : bool, default False
            Draw half circles if bus_sizes is a pandas.Series with a Multiindex.
            If set to true, the upper half circle per bus then includes all positive values
            of the series, the lower half circle all negative values. Defaults to False.
        bus_colors : str/dict/pandas.Series
            Colors for the buses, defaults to "cadetblue". If bus_sizes is a
            pandas.Series with a Multiindex, bus_colors defaults to the
            n.carriers['color'] column.
        bus_cmap : mcolors.Colormap/str
            If bus_colors are floats, this color map will assign the colors
        bus_cmap_norm : plt.Normalize
            The norm applied to the bus_cmap
        bus_alpha : float/dict/pandas.Series
            Adds alpha channel to buses, defaults to 1
        geometry : bool, default False
            Whether to use the geometry column of the branch components
        flow : Callable
            Function to calculate the flow for each branch component.
        auto_scale_branches : bool
            Whether to auto scale the flow and branch sizes. If true, the function
            uses a rough auto-scaling when plotting flows as well as it creates
            a LineCollection with widths in data units. If false, the function
            does not scale the flow and branch sizes and creates a PatchCollection
            with widths in display units. The latter is useful for plotting
            consistent branch widths and flows in different zoom levels.

        Other Parameters
        ----------------
        line_flow : str/callable/dict/pandas.Series/Network.snapshots, default None
            Flow to be for each line branch. If an element of
            n.snapshots is given, the flow at this timestamp will be
            displayed. If an aggregation function is given, is will be applied
            to the total network flow via pandas.DataFrame.agg (accepts also
            function names). Otherwise flows can be specified by passing a pandas
            Series. Use the corresponding width argument to adjust size of the
            flow arrows.
        line_colors : str/pandas.Series
            Colors for the lines, defaults to 'rosybrown'.
        line_cmap : mcolors.Colormap/str|dict
            If line_colors are floats, this color map will assign the colors.
        line_cmap_norm : plt.Normalize
            The norm applied to the line_cmap.
        line_alpha : str/pandas.Series
            Alpha for the lines, defaults to 1.
        line_widths : dict/pandas.Series
            Widths of lines, defaults to 1.5
        link_flow : str/callable/dict/pandas.Series/Network.snapshots, default None
            Flow to be for each link branch. See line_flow for more information.
        link_colors : str/pandas.Series
            Colors for the links, defaults to 'darkseagreen'.
        link_cmap : mcolors.Colormap/str|dict
            If link_colors are floats, this color map will assign the colors.
        link_cmap_norm : plt.Normalize|matplotlib.colors.*Norm
            The norm applied to the link_cmap.
        link_alpha : str/pandas.Series
            Alpha for the links, defaults to 1.
        link_widths : dict/pandas.Series
            Widths of links, defaults to 1.5
        transformer_flow : str/callable/dict/pandas.Series/Network.snapshots, default None
            Flow to be for each transformer branch. See line_flow for more information.
        transformer_colors : str/pandas.Series
            Colors for the transfomer, defaults to 'orange'.
        transformer_cmap : mcolors.Colormap/str|dict
            If transformer_colors are floats, this color map will assign the colors.
        transformer_cmap_norm : matplotlib.colors.Normalize|matplotlib.colors.*Norm
            The norm applied to the transformer_cmap.
        transformer_alpha : str/pandas.Series
            Alpha for the transfomer, defaults to 1.
        transformer_widths : dict/pandas.Series
            Widths of transformer, defaults to 1.5

        .. deprecated:: 0.28.0
            `flow` will be deprecated, use `line_flow`, `link_flow` and `transformer_flow`
                instead. The argument will be passed to all branches.
            `bus_norm`, `line_norm`, `link_norm` and `transformer_norm` are deprecated,
                use `bus_cmap_norm`, `line_cmap_norm`, `link_cmap_norm` and
                `transformer_cmap_norm` instead.
            `color_geomap` is deprecated, use `geomap_colors` instead.

        Returns
        -------
        dict: collections of matplotlib objects
            2D dictinary with the following keys:
            - 'nodes'
                - 'Bus': Collection of bus points
            - 'branches' (for each branch component)
                - 'lines': Collection of line branches
                - 'links': Collection of link branches
                - 'transformers': Collection of transformer branches
            - 'flows' (for each branch component)
                - 'lines': Collection of line flows
                - 'links': Collection of link flows
                - 'transformers': Collection of transformer flows

        """
        n = self.n

        self.init_axis(
            ax=ax,
            projection=projection,
            geomap=geomap,
            geomap_resolution=geomap_resolution,
            geomap_colors=geomap_colors,
            title=title,
            boundaries=boundaries,
        )

        if self.ax is None:
            msg = "No axis passed or created."
            raise ValueError(msg)

        if flow is not None:
            if (
                line_flow is not None
                or link_flow is not None
                or transformer_flow is not None
            ):
                msg = "The `flow` argument is deprecated, use `line_flow`, `link_flow` and "
                "`transformer_flow` instead. You can't use both arguments at the same time."
                raise ValueError(msg)
            if isinstance(flow, pd.Series) and isinstance(flow.index, pd.MultiIndex):
                msg = (
                    "The `flow` argument is deprecated, use `line_flow`, `link_flow` and "
                    "`transformer_flow` instead. Multiindex Series are not supported anymore."
                    "Deprecated in version 0.34 and will be removed in version 1.0."
                )
                warnings.warn(msg, DeprecationWarning, 2)
                line_flow = flow.get("Line")
                link_flow = flow.get("Link")
                transformer_flow = flow.get("Transformer")

        # Deprecation errors
        if isinstance(line_widths, pd.Series) and isinstance(
            line_widths.index, pd.MultiIndex
        ):
            msg = (
                "Index of argument 'line_widths' is a Multiindex, "
                "this is not support since pypsa v0.17 and will be removed in v1.0. "
                "Set differing widths with arguments 'line_widths', "
                "'link_widths' and 'transformer_widths'."
            )
            raise DeprecationWarning(msg)

        if isinstance(line_colors, pd.Series) and isinstance(
            line_colors.index, pd.MultiIndex
        ):
            msg = (
                "Index of argument 'line_colors' is a Multiindex, "
                "this is not support since pypsa v0.17 and will be removed in v1.0. "
                "Set differing colors with arguments 'line_colors', "
                "'link_colors' and 'transformer_colors'."
            )
            raise DeprecationWarning(msg)

        # Check for ValueErrors
        if geomap and not cartopy_present:
            logger.warning("Cartopy needs to be installed to use `geomap=True`.")
            geomap = False

        # Check if bus_sizes is a MultiIndex
        multindex_buses = isinstance(bus_sizes, pd.Series) and isinstance(
            bus_sizes.index, pd.MultiIndex
        )

        # Apply default values
        if bus_colors is None:
            if multindex_buses:
                bus_colors = n.carriers.color
            else:
                bus_colors = "cadetblue"

        # Format different input types
        bus_colors = _convert_to_series(bus_colors, n.buses.index)
        bus_sizes = _convert_to_series(bus_sizes, n.buses.index)

        # Add missing colors
        # TODO: This is not consistent, since for multiindex a ValueError is raised
        if not multindex_buses:
            bus_colors = bus_colors.reindex(n.buses.index)

        # Raise additional ValueErrors after formatting
        if multindex_buses:
            if len(bus_sizes.index.unique(level=0).difference(n.buses.index)) != 0:
                msg = "The first MultiIndex level of sizes must contain buses"
                raise ValueError(msg)
            if not bus_sizes.index.unique(level=1).isin(bus_colors.index).all():
                msg = "Colors not defined for all elements in the second MultiIndex "
                "level of sizes, please make sure that all the elements are "
                "included in colors or in n.carriers.color"
                raise ValueError(msg)

        # Apply all cmaps
        bus_colors = _apply_cmap(bus_colors, bus_cmap, bus_cmap_norm)

        # Plot buses
        bus_sizes = bus_sizes.sort_index(level=0, sort_remaining=False)
        if geomap:
            bus_sizes = bus_sizes * self.area_factor**2
        if isinstance(bus_sizes.index, pd.MultiIndex):
            patches: list[Circle] | list[Wedge]
            patches = self.get_multiindex_buses(
                bus_sizes, bus_colors, bus_alpha, bus_split_circles
            )
        else:
            patches = self.get_singleindex_buses(bus_sizes, bus_colors, bus_alpha)
        bus_collection = PatchCollection(patches, match_original=True, zorder=5)
        self.ax.add_collection(bus_collection)

        # Plot branches and flows
        if branch_components is None:
            branch_components = n.branch_components

        branch_collections = {}
        flow_collections = {}

        for c in n.iterate_components(branch_components):
            # Get branch collection
            if c.name == "Line":
                widths = line_widths
                colors = line_colors
                alpha = line_alpha
                flow = self._flow_ds_from_arg(line_flow, c.name)
                cmap = line_cmap
                cmap_norm = line_cmap_norm
            elif c.name == "Link":
                widths = link_widths
                colors = link_colors
                alpha = link_alpha
                flow = self._flow_ds_from_arg(link_flow, c.name)
                cmap = link_cmap
                cmap_norm = link_cmap_norm
            elif c.name == "Transformer":
                widths = transformer_widths
                colors = transformer_colors
                alpha = transformer_alpha
                flow = self._flow_ds_from_arg(transformer_flow, c.name)
                cmap = transformer_cmap
                cmap_norm = transformer_cmap_norm

            data = self._dataframe_from_arguments(
                c.static.index, widths=widths, colors=colors, alpha=alpha, flow=flow
            )
            if data.empty:
                continue

            data["colors"] = _apply_cmap(data.colors, cmap, cmap_norm)

            branch_coll = self.get_branch_collection(
                c, data.widths, data.colors, data.alpha, geometry, auto_scale_branches
            )
            if branch_coll is not None:
                self.ax.add_collection(branch_coll)
                branch_collections[c.name] = branch_coll

            # Get flow collection if flow data exists
            if flow is not None:
                if auto_scale_branches:
                    rough_scale = (
                        sum([len(n.static(c)) for c in branch_components]) + 100
                    )
                    data["flow"] = (
                        data.flow.mul(abs(data.widths), fill_value=0) / rough_scale
                    )
                flow_coll = self.get_flow_collection(
                    c,
                    flow=data.flow,
                    widths=data.widths,
                    colors=data.colors,
                    alpha=data.alpha,
                )
                if flow_coll is not None:
                    self.ax.add_collection(flow_coll)
                    flow_collections[c.name] = flow_coll

        return {
            "nodes": {"Bus": bus_collection},
            "branches": branch_collections,
            "flows": flow_collections,
        }


@deprecated_kwargs(
    bus_norm="bus_cmap_norm",
    line_norm="line_cmap_norm",
    link_norm="link_cmap_norm",
    transformer_norm="transformer_cmap_norm",
    color_geomap="geomap_colors",
    deprecated_in="0.34",
    removed_in="1.0",
)
@wraps(
    MapPlotter.draw_map,
    assigned=("__doc__", "__annotations__", "__type_params__"),
)
def plot(  # noqa: D103
    n: Network,
    ax: Axes | None = None,
    layouter: Callable | None = None,
    boundaries: tuple[float, float, float, float] | None = None,
    margin: float | None = 0.05,
    projection: Any = None,
    geomap: bool | str = True,
    geomap_resolution: Literal["10m", "50m", "110m"] = "50m",
    geomap_colors: dict | bool | None = None,
    title: str = "",
    jitter: float | None = None,
    **kwargs: Any,
) -> dict:
    if margin is None:
        logger.warning(
            "The `margin` argument does support None value anymore. "
            "Falling back to the default value 0.05. This will raise "
            "an error in the future."
        )
        margin = 0.05

    bus_sizes = kwargs.get("bus_sizes")
    multindex_buses = isinstance(bus_sizes, pd.Series) and isinstance(
        bus_sizes.index, pd.MultiIndex
    )
    if isinstance(bus_sizes, pd.Series):
        buses = (
            bus_sizes.index if not multindex_buses else bus_sizes.index.unique(level=0)
        )
    else:
        buses = n.buses.index

    if isinstance(geomap, str):
        logger.warning(
            "The `geomap` argument now only accepts a boolean value. "
            "If you want to set the resolution, use the `geomap_resolution` "
            "argument instead."
        )
        geomap = True
        geomap_resolution = geomap  # type: ignore

    # setup plotter
    plotter = MapPlotter(
        n,
        layouter,
        boundaries=boundaries,
        margin=margin,
        buses=buses,
    )

    # Add jitter if given
    if jitter is not None:
        plotter.add_jitter(jitter)

    return plotter.draw_map(
        ax,
        projection=projection,
        geomap=geomap,
        geomap_resolution=geomap_resolution,
        geomap_colors=geomap_colors,
        title=title,
        **kwargs,
    )


class HandlerCircle(HandlerPatch):
    """Legend Handler used to create circles for legend entries.

    This handler resizes the circles in order to match the same
    dimensional scaling as in the applied axis.
    """

    LEGEND_SCALE_FACTOR = 72

    def __init__(self, scale_factor: float | None = None) -> None:
        """Initialize the HandlerCircle."""
        super().__init__()
        self.scale_factor = scale_factor or self.LEGEND_SCALE_FACTOR

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Circle,  # type: ignore
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any,
    ) -> list[Circle]:
        """Create the artists for the legend."""
        fig = legend.get_figure()
        if fig is None:
            msg = "Legend must be placed on a figure. No figure found."
            raise ValueError(msg)

        ax = legend.axes

        # take minimum to protect against too uneven x- and y-axis extents
        unit = min(np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0])
        norm = (self.scale_factor / fig.dpi) * unit
        radius = orig_handle.get_radius() * norm
        center = 5 - xdescent, 3 - ydescent
        p = plt.Circle(center, radius)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class WedgeHandler(HandlerPatch):
    """Legend Handler used to create sermi-circles for legend entries.

    This handler resizes the semi-circles in order to match the same
    dimensional scaling as in the applied axis.
    """

    LEGEND_SCALE_FACTOR = 72

    def __init__(self, scale_factor: float | None = None) -> None:
        """Initialize the WedgeHandler."""
        super().__init__()
        self.scale_factor = scale_factor or self.LEGEND_SCALE_FACTOR

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Wedge,  # type: ignore
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any,
    ) -> list[Wedge]:
        """Create the artists for the legend."""
        fig = legend.get_figure()
        if fig is None:
            msg = "Legend must be placed on a figure. No figure found."
            raise ValueError(msg)
        ax = legend.axes
        center = 5 - xdescent, 3 - ydescent
        unit = min(np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0])
        norm = (self.scale_factor / fig.dpi) * unit
        r = orig_handle.r * norm
        p = Wedge(
            center=center,
            r=r,
            theta1=orig_handle.theta1,
            theta2=orig_handle.theta2,
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerArrow(HandlerPatch):
    """Handler for FancyArrow patches in legends."""

    # Empirically determined scale factor for legend arrow sizes
    LEGEND_SCALE_FACTOR = 72

    def __init__(
        self, width_ratio: float = 0.2, scale_factor: float | None = None
    ) -> None:
        """Initialize the HandlerArrow.

        Parameters
        ----------
        width_ratio : float, optional
            Ratio of arrow width to head width, by default 0.2
        scale_factor : float, optional
            Custom scaling factor for arrow size, by default None

        """
        super().__init__()
        self.width_ratio = width_ratio
        self.scale_factor = scale_factor or self.LEGEND_SCALE_FACTOR

    def create_artists(
        self,
        legend: Legend,
        orig_handle: FancyArrow,  # type: ignore
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any,
    ) -> list[FancyArrow]:
        """Create the artists for the legend."""
        fig = legend.get_figure()
        if fig is None:
            msg = "Legend must be placed on a figure. No figure found."
            raise ValueError(msg)
        ax = legend.axes
        unit = min(np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0])
        norm = (self.scale_factor / fig.dpi) * unit
        arrow = FancyArrow(
            0,
            ydescent + height / 2,
            width / 4,
            0,
            head_width=orig_handle._head_width * norm,  # type: ignore
            head_length=orig_handle._head_length * norm,  # type: ignore
            length_includes_head=False,
            width=orig_handle._head_width * self.width_ratio * norm,  # type: ignore
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_facecolor(),
            **{k: getattr(orig_handle, f"get_{k}")() for k in ["linewidth", "alpha"]},
        )
        return [arrow]


def add_legend_lines(
    ax: Axes,
    sizes: list[float] | np.ndarray,
    labels: list[str] | np.ndarray,
    colors: list[str] | np.ndarray | None = None,
    patch_kw: dict[str, Any] | None = None,
    legend_kw: dict[str, Any] | None = None,
) -> Legend:
    """Add a legend for lines and links.

    Parameters
    ----------
    ax : matplotlib ax
        Matplotlib axis to add the legend to.
    sizes : list-like, float
        Size of the line reference; for example [3, 2, 1]
    labels : list-like, str
        Label of the line reference; for example ["30 GW", "20 GW", "10 GW"]
    colors: list-like, str
        Color of the line reference; for example ["red, "green", "blue"]
    patch_kw : defaults to {}
        Keyword arguments passed to plt.Line2D
    legend_kw : defaults to {}
        Keyword arguments passed to ax.legend

    """
    sizes = np.atleast_1d(sizes)
    labels = np.atleast_1d(labels)
    colors = [] if colors is None else np.atleast_1d(colors)
    if patch_kw is None:
        patch_kw = {}
    if legend_kw is None:
        legend_kw = {}

    if len(sizes) != len(labels):
        msg = "Sizes and labels must have the same length."
        raise ValueError(msg)
    if len(colors) > 0 and len(sizes) != len(colors):
        msg = "Sizes, labels, and colors must have the same length."
        raise ValueError(msg)

    if len(colors) == 0:
        handles = [plt.Line2D([0], [0], linewidth=s, **patch_kw) for s in sizes]
    else:
        handles = [
            plt.Line2D([0], [0], linewidth=s, color=c, **patch_kw)
            for s, c in zip(sizes, colors, strict=False)
        ]

    legend = ax.legend(handles, labels, **legend_kw)

    fig = ax.get_figure()
    if fig is not None:
        fig.add_artist(legend)

    return legend


def add_legend_patches(
    ax: Axes,
    colors: list[str] | np.ndarray,
    labels: list[str] | np.ndarray,
    patch_kw: dict[str, Any] | None = None,
    legend_kw: dict[str, Any] | None = None,
) -> Legend:
    """Add patches for color references.

    Parameters
    ----------
    ax : matplotlib ax
        Matplotlib axis to add the legend to.
    colors : list-like, float
        Color of the patch; for example ["r", "g", "b"]
    labels : list-like, str
        Label of the patch; for example ["wind", "solar", "gas"]
    patch_kw : defaults to {}
        Keyword arguments passed to matplotlib.patches.Patch
    legend_kw : defaults to {}
        Keyword arguments passed to ax.legend

    """
    colors = np.atleast_1d(colors)
    labels = np.atleast_1d(labels)
    if patch_kw is None:
        patch_kw = {}
    if legend_kw is None:
        legend_kw = {}

    if len(colors) != len(labels):
        msg = "Colors and labels must have the same length."
        raise ValueError(msg)

    handles = [Patch(facecolor=c, **patch_kw) for c in colors]

    legend = ax.legend(handles, labels, **legend_kw)

    fig = ax.get_figure()
    if fig is not None:
        fig.add_artist(legend)

    return legend


def add_legend_circles(
    ax: Axes,
    sizes: list[float] | np.ndarray,
    labels: list[str] | np.ndarray,
    srid: int = DEFAULT_EPSG,
    patch_kw: dict[str, Any] | None = None,
    legend_kw: dict[str, Any] | None = None,
) -> Legend:
    """Add a legend for reference circles.

    .. warning::
        When combining ``n.plot()`` with other plots on a geographical axis,
        ensure ``n.plot()`` is called first or the final axis extent is set initially
        (``ax.set_extent(boundaries, crs=crs)``) for consistent legend circle sizes.

    Parameters
    ----------
    ax : matplotlib ax
        Matplotlib axis to add the legend to.
    sizes : list-like, float
        Size of the reference circle; for example [3, 2, 1]
    labels : list-like, str
        Label of the reference circle; for example ["30 GW", "20 GW", "10 GW"]
    srid : int, defaults to DEFAULT_EPSG
        Spatial reference ID for area correction
    patch_kw : defaults to {}
        Keyword arguments passed to matplotlib.patches.Circle
    legend_kw : defaults to {}
        Keyword arguments passed to ax.legend

    """
    sizes = np.atleast_1d(sizes)
    labels = np.atleast_1d(labels)
    if patch_kw is None:
        patch_kw = {}
    if legend_kw is None:
        legend_kw = {}

    if len(sizes) != len(labels):
        msg = "Sizes and labels must have the same length."
        raise ValueError(msg)

    if hasattr(ax, "projection"):
        warnings.warn(
            "When combining n.plot() with other plots on a geographical axis, "
            "ensure n.plot() is called first or the final axis extent is set initially "
            "(ax.set_extent(boundaries, crs=crs)) for consistent legend circle sizes.",
            UserWarning,
            stacklevel=2,
        )
        area_correction = get_projected_area_factor(ax, srid) ** 2
        sizes = [s * area_correction for s in sizes]

    handles = [Circle((0, 0), radius=s**0.5, **patch_kw) for s in sizes]

    legend = ax.legend(
        handles, labels, handler_map={Circle: HandlerCircle()}, **legend_kw
    )

    fig = ax.get_figure()
    if fig is not None:
        fig.add_artist(legend)

    return legend


def add_legend_semicircles(
    ax: Axes,
    sizes: list[float] | np.ndarray,
    labels: list[str] | np.ndarray,
    srid: int = DEFAULT_EPSG,
    patch_kw: dict[str, Any] | None = None,
    legend_kw: dict[str, Any] | None = None,
) -> Legend:
    """Add a legend for reference semi-circles.

    .. warning::
        When combining ``n.plot()`` with other plots on a geographical axis,
        ensure ``n.plot()`` is called first or the final axis extent is set initially
        (``ax.set_extent(boundaries, crs=crs)``) for consistent legend semicircle sizes.

    Parameters
    ----------
    ax : matplotlib ax
        Matplotlib axis to add the legend to.
    sizes : list-like, float
        Size of the reference circle; for example [3, 2, 1]
    labels : list-like, str
        Label of the reference circle; for example ["30 GW", "20 GW", "10 GW"]
    srid : int, default 4326
        Spatial reference ID for area correction
    patch_kw : defaults to {}
        Keyword arguments passed to matplotlib.patches.Wedges
    legend_kw : defaults to {}
        Keyword arguments passed to ax.legend

    """
    sizes = np.atleast_1d(sizes)
    labels = np.atleast_1d(labels)

    if len(sizes) != len(labels):
        msg = "Sizes and labels must have the same length."
        raise ValueError(msg)

    if hasattr(ax, "projection"):
        warnings.warn(
            "When combining n.plot() with other plots on a geographical axis, "
            "ensure n.plot() is called first or the final axis extent is set initially "
            "(ax.set_extent(boundaries, crs=crs)) for consistent legend semicircle sizes.",
            UserWarning,
            stacklevel=2,
        )
        area_correction = get_projected_area_factor(ax, srid) ** 2
        sizes = [s * area_correction for s in sizes]

    if patch_kw is None:
        patch_kw = {}
    if legend_kw is None:
        legend_kw = {}

    radius = [np.sign(s) * np.abs(s * 2) ** 0.5 for s in sizes]
    handles = [
        Wedge((0, -r / 2), r=r, theta1=0, theta2=180, **patch_kw) for r in radius
    ]

    legend = ax.legend(
        handles, labels, handler_map={Wedge: WedgeHandler()}, **legend_kw
    )

    fig = ax.get_figure()
    if fig is not None:
        fig.add_artist(legend)

    return legend


def add_legend_arrows(
    ax: Axes,
    sizes: list[float] | np.ndarray,
    labels: list[str] | np.ndarray,
    srid: int = 4326,
    colors: list[str] | np.ndarray | None = None,
    arrow_to_tail_width: float = 0.15,
    patch_kw: dict[str, Any] | None = None,
    legend_kw: dict[str, Any] | None = None,
) -> Legend:
    """Add a legend for flow arrows.

    Parameters
    ----------
    ax : matplotlib ax
        Matplotlib axis to add the legend to.
    sizes : list-like, float
        Size of the reference arrows; for example [3, 2, 1]
    labels : list-like, str
        Label of the reference arrows; for example ["30 GW", "20 GW", "10 GW"]
    srid : int, default 4326
        Spatial reference ID for area correction
    colors : str/list-like, default 'b'
        Color(s) of the arrows
    arrow_to_tail_width : float, default 0.15
        Ratio of arrow width to tail width
    patch_kw : dict, optional
        Keyword arguments passed to FancyArrow
    legend_kw : dict, optional
        Keyword arguments passed to ax.legend

    """
    sizes = np.atleast_1d(sizes) ** 0.5
    labels = np.atleast_1d(labels)
    colors = np.atleast_1d(colors)  # type: ignore

    if patch_kw is None:
        patch_kw = {"linewidth": 1, "zorder": 4}
    if legend_kw is None:
        legend_kw = {}

    if len(sizes) != len(labels):
        msg = "Sizes and labels must have the same length."
        raise ValueError(msg)

    if hasattr(ax, "projection"):
        area_correction = get_projected_area_factor(ax, srid)
        sizes = [s * area_correction for s in sizes]

    if len(colors) == 1:
        colors = np.repeat(colors, len(sizes))
    elif len(colors) != len(sizes):
        msg = "Colors must be a single value or match length of sizes"
        raise ValueError(msg)

    # Scale sizes to be more visible in legend
    handles = [
        FancyArrow(
            0,
            0,
            1,
            0,  # Shorter arrow length
            head_width=s,
            head_length=s / 0.6,
            length_includes_head=False,
            color=c,
            **patch_kw,
        )
        for s, c in zip(sizes, colors, strict=False)
    ]

    legend = ax.legend(
        handles,
        labels,
        handler_map={FancyArrow: HandlerArrow(width_ratio=arrow_to_tail_width)},
        **legend_kw,
    )
    fig = ax.get_figure()
    if fig is not None:
        fig.add_artist(legend)

    return legend


def round_to_significant_digits(x: float, n: int = 2) -> int | float:
    """Round a number to n significant figures."""
    if x == 0:
        return 0
    magnitude = int(np.floor(np.log10(abs(x))))
    rounded = round(x, -magnitude + (n - 1))
    return int(rounded) if rounded >= 1 else rounded


def scaled_legend_label(value: float, base_unit: str = "MWh") -> str:
    """Scale a value to an appropriate unit for legend labels.

    This function scales the value to a more human-readable format. Ensures scaled
    values >= 1 are integers.
    """
    unit_scales = {
        "": 1,  # base
        "k": 1e3,  # kilo
        "M": 1e6,  # mega
        "G": 1e9,  # giga
        "T": 1e12,  # tera
        "P": 1e15,  # peta
    }

    # Extract base unit without prefix
    base_prefix = ""
    unit_name = base_unit
    for prefix in sorted(unit_scales.keys(), key=len, reverse=True):
        if base_unit.startswith(prefix):
            base_prefix = prefix
            unit_name = base_unit[len(prefix) :]
            break

    # Calculate absolute value in base units
    base_value = value * unit_scales[base_prefix]

    # Find appropriate prefix
    magnitude = np.floor(np.log10(abs(base_value))) if base_value != 0 else 0

    # Get closest unit scale that keeps value between 1 and 1000
    scales = np.array(list(unit_scales.values()))
    prefixes = list(unit_scales.keys())
    target_scale_idx = np.searchsorted(scales, 10 ** (magnitude - 2))
    if target_scale_idx >= len(scales):
        target_scale_idx = len(scales) - 1

    target_scale = scales[target_scale_idx]
    target_prefix = prefixes[target_scale_idx]

    # If base_unit already has a prefix, adjust the scale accordingly
    if base_prefix:
        # Calculate the relative scale between target and base prefix
        scale_difference = target_scale / unit_scales[base_prefix]
        scaled_value = value / scale_difference
    else:
        scaled_value = base_value / target_scale

    # Convert to integer if >= 1
    if abs(scaled_value) >= 1:
        scaled_value = int(round(scaled_value))

    return f"{scaled_value} {target_prefix}{unit_name}"


def get_legend_representatives(
    series: pd.Series,
    quantiles: list[float] | None = None,
    n_significant: int = 1,
    base_unit: str = "MWh",
    group_on_first_level: bool = False,
) -> list[tuple[int | float, str]]:
    """Get representative values from a numeric series for legend visualization.

    Automatic unit scaling is applied. Values >= 1 are returned as integers.

    Parameters
    ----------
    series : pd.Series
        Series containing the values
    quantiles : list
        List of quantile to use assuming a uniform distribution from
        0 to the maximum value (default: [0.6, 0.2])
    n_significant : int
        Number of significant figures to round to
    base_unit : str
        Base unit of the values (default: "MWh")
    group_on_first_level : bool
        If True, group the series by the first level of the index
        before calculating the maximum value

    Returns
    -------
    list
        List of tuples (scaled_value, unit) for each quantile

    """
    if quantiles is None:
        quantiles = [0.6, 0.2]
    if series.empty:
        return []
    if group_on_first_level:
        series = series.abs().groupby(level=0).sum()
    max_value = series.abs().max()
    values = [max_value * q for q in quantiles]
    rounded_values = [round_to_significant_digits(v, n_significant) for v in values]

    return [(v, scaled_legend_label(v, base_unit)) for v in rounded_values]
