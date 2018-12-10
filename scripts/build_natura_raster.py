import atlite
from osgeo import gdal
import geokit as gk

def determine_cutout_xXyY(cutout_name):
    cutout = atlite.Cutout(cutout_name, cutout_dir="../cutouts")
    x, X, y, Y = cutout.extent
    dx = (X - x) / (cutout.shape[1] - 1)
    dy = (Y - y) / (cutout.shape[0] - 1)
    return [x - dx/2., X + dx/2., y - dy/2., Y + dy/2.]

cutout_names = np.unique([res['cutout'] for res in config['renewable'].values()])
xs, Xs, ys, Ys = zip(*(determine_cutout_xyXY(cutout) for cutout in cutout_names))
xXyY = min(xs), max(Xs), min(ys), max(Ys)

natura = gk.vector.loadVector(snakemake.input[0])
extent = gk.Extent.from_xXyY(xXyY).castTo(3035).fit(100)
extent.rasterize(natura, pixelWidth=100, pixelHeight=100, output=snakemake.output[0])
