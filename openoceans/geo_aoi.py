import geopandas as gpd
from shapely.geometry import Polygon, mapping
import simplekml
import warnings
from ipyleaflet import Map, basemaps, DrawControl
from ipyleaflet import Polygon as IPyPolygon
from IPython.display import display

class GeoAOI:
    # stores a bounding box and has basic io/viz methods
    def __init__(self, poly=None):
        self.poly = poly

    def edit(self):

        # check that we're in a notebook/ipython environment
        is_nb = self._is_notebook()

        if is_nb == False:
            warnings.warn(
                'Interactively editing bounding box only available in notebook/qtconsole environments.')
            return None

        sp = self.poly  # shapely polygon of bounding box
        centroid_xy = self.poly.centroid.coords.xy
        # ipyleaflet uses lat/long here, because ???
        plot_center = (centroid_xy[1][0], centroid_xy[0][0])
        collection = []

        def handle_draw(target, action, geo_json):

            if action == 'edited':
                self.poly = Polygon(geo_json['geometry']['coordinates'][0])

        m = Map(basemap=basemaps.Esri.WorldImagery,
                center=plot_center, zoom=7, scroll_wheel_zoom=True)
        dc = DrawControl(circlemarker={}, rectangle={},
                         polyline={}, polygon={}, edit=True, remove=False)

        dc.data = [{'type': 'Feature',
                   'properties':
                    {'style': {'stroke': True,
                               'color': '#bf5700',
                               'weight': 4,
                               'opacity': 0.5,
                               'fill': True,
                               'fillColor': '#bf5700',
                               'fillOpacity': 0.5,
                               'clickable': True}},
                    'geometry': mapping(sp)}]

        dc.on_draw(handle_draw)
        m.add_control(dc)
        display(m)

    def to_kml(self, filepath, polygon=None, polygon_name='Bounding Box'):
        '''Saves bbox as a kml for easy viewing in Google Earth.'''
        kml = simplekml.Kml()
        pol = kml.newpolygon(name=polygon_name)

        pol.outerboundaryis = list(zip(self.poly.exterior.coords.xy[0],
                                       self.poly.exterior.coords.xy[1]))

        pol.style.linestyle.color = simplekml.Color.red
        pol.style.linestyle.width = 3
        pol.style.polystyle.color = simplekml.Color.changealphaint(
            100, simplekml.Color.red)
        kml.save(filepath)

    def to_geojson(self, filepath=None):
        geos = gpd.GeoSeries([self.poly])
        if filepath == None:
            return geos.__geo_interface__
        else:
            geos.to_file(filepath, driver='GeoJSON')

    def explore(self):
        '''Formats bbox polygon as a geodataframe and calls the explore() function.'''

        # need to update to use imagery basemap
        gdf = gpd.GeoDataFrame(geometry=[self.poly], crs="EPSG:4326")
        return gdf.explore()

    @classmethod
    def from_geojson(cls, filepath):
        '''Constucts a geodataframe from a geojson file, presumed to only contain a single polygon.'''
        gdf = gpd.read_file(filepath)
        if gdf.shape[0] > 1:
            warnings.warn('Multiple polygons detected! Using first in stack.')

        p = gdf.geometry.iloc[0]

        return cls(p)

    @classmethod
    def from_points(cls, x, y):
        '''Constructs a polygon from arrays of longitude and latitude points and 
        stores as a geodataframe.'''
        p = Polygon(list(zip(x, y)))
        return cls(p)

    @classmethod
    def from_drawing(cls):

        bb = cls()

        # check that we're in a notebook/ipython environment
        is_nb = _is_notebook()

        if is_nb == False:
            warnings.warn(
                'Interactively drawn bounding box only available in notebook/qtconsole environments.')
            return None

        collection = []

        def handle_draw(target, action, geo_json):

            # format drawn bbox as shapely polygon
            draw_item = Polygon(geo_json['geometry']['coordinates'][0])

            if action == 'created':
                collection.append(draw_item)

                if len(collection) > 1:
                    warnings.warn(
                        'Only 1 polygon allowed for bounding box. Use delete tool to remove extra boxes and click save before continuing.')

            if action == 'deleted':
                collection.remove(draw_item)

            bb.poly = collection[0]

        m = Map(basemap=basemaps.Esri.WorldImagery, center=(
            23.951465, -78.121304), zoom=4, scroll_wheel_zoom=True)
        dc = DrawControl(circlemarker={}, polyline={}, polygon={}, edit=False)

        dc.rectangle = {
            "shapeOptions": {
                "fillColor": "#bf5700",
                "color": "#bf5700",
                "fillOpacity": 0.5
            }
        }

        dc.polygon = {
            "shapeOptions": {
                "fillColor": "#bf5700",
                "color": "#bf5700",
                "fillOpacity": 0.5
            }
        }

        dc.on_draw(handle_draw)
        m.add_control(dc)
        display(m)
        return bb


if __name__ == "__main__":
    from utils import _is_notebook

else:
    from .utils import _is_notebook

