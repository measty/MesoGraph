from cmath import pi
from xyzservices import TileProvider
from bokeh.plotting import figure, output_file, show
from bokeh.models.tiles import WMTSTileSource, TMSTileSource, TileSource, MercatorTileSource
from bokeh.core.properties import Bool, Override
from bokeh.util.compiler import JavaScript, TypeScript
from bokeh.models import Slider, Toggle, Dropdown, PreText
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import layout
from pathlib import Path
import json
import numpy as np
from bokeh.plotting import ColumnDataSource
from bokeh.models import GeoJSONDataSource
from bokeh.transform import linear_cmap

#import {TileSource} from "C:/Users/meast/anaconda3/envs/bkdev/Lib/site-packages/bokeh-3.0.0.dev1+47.g281e3f87d-py3.9.egg/bokeh/server/static/js/lib/models/tiles/tile_source"
#import * as p from "C:/Users/meast/anaconda3/envs/bkdev/Lib/site-packages/bokeh-3.0.0.dev1+47.g281e3f87d-py3.9.egg/bokeh/server/static/js/lib/core/properties"
#import {range} from "C:/Users/meast/anaconda3/envs/bkdev/Lib/site-packages/bokeh-3.0.0.dev1+47.g281e3f87d-py3.9.egg/bokeh/server/static/js/lib/core/util/array"
#import {Extent, Bounds, meters_extent_to_geographic} from "C:/Users/meast/anaconda3/envs/bkdev/Lib/site-packages/bokeh-3.0.0.dev1+47.g281e3f87d-py3.9.egg/bokeh/server/static/js/lib/models/tiles/tile_utils"

TS_CODE = '''

import {TileSource} from "models/tiles/tile_source"
import * as p from "core/properties"
import {range} from "core/util/array"
import {Extent, Bounds, meters_extent_to_geographic} from "models/tiles/tile_utils"

export namespace MercatorTileSource {
  export type Attrs = p.AttrsOf<Props>

  export type Props = TileSource.Props & {
    snap_to_zoom: p.Property<boolean>
    wrap_around: p.Property<boolean>
  }
}

export interface MercatorTileSource extends MercatorTileSource.Attrs {}

export class MercatorTileSource extends TileSource {
  override properties: MercatorTileSource.Props

  constructor(attrs?: Partial<MercatorTileSource.Attrs>) {
    super(attrs)
  }

  static {
    this.define<MercatorTileSource.Props>(({Boolean}) => ({
      snap_to_zoom: [ Boolean, false ],
      wrap_around:  [ Boolean, false ],
    }))

    this.override<MercatorTileSource.Props>({
      x_origin_offset:    0,
      y_origin_offset:    0,
      initial_resolution: 15654,
      tile_size: 254
    })
  }

  protected _resolutions: number[]

  override initialize(): void {
    super.initialize()
    this._resolutions = range(this.min_zoom, this.max_zoom+1).map((z) => this.get_resolution(z))
  }

  protected _computed_initial_resolution(): number {
    if (this.initial_resolution != null)
      return this.initial_resolution
    else {
      // TODO testing 2015-11-17, if this codepath is used it seems
      // to use 100% cpu and wedge Chrome
      return (2 * Math.PI * 6378137) / this.tile_size
    }
  }

  is_valid_tile(x: number, y: number, z: number): boolean {
    if (!this.wrap_around) {
      if (x < 0 || x >= 2**z)
        return false
    }

    if (y < 0 || y >= 2**z)
      return false

    return true
  }

  parent_by_tile_xyz(x: number, y: number, z: number): [number, number, number] {
    const quadkey = this.tile_xyz_to_quadkey(x, y, z)
    const parent_quadkey = quadkey.substring(0, quadkey.length - 1)
    return this.quadkey_to_tile_xyz(parent_quadkey)
  }

  get_resolution(level: number): number {
    return this._computed_initial_resolution() / 2**level
  }

  get_resolution_by_extent(extent: Extent, height: number, width: number): [number, number] {
    const x_rs = (extent[2] - extent[0]) / width
    const y_rs = (extent[3] - extent[1]) / height
    return [x_rs, y_rs]
  }

  get_level_by_extent(extent: Extent, height: number, width: number): number {
    const x_rs = (extent[2] - extent[0]) / width
    const y_rs = (extent[3] - extent[1]) / height
    const resolution = Math.max(x_rs, y_rs)

    let i = 0
    for (const r of this._resolutions) {
      if (resolution > r) {
        if (i == 0)
          return 0
        if (i > 0)
          return i - 1
      }
      i += 1
    }

    // otherwise return the highest available resolution
    return (i-1)
  }

  get_closest_level_by_extent(extent: Extent, height: number, width: number): number {
    const x_rs = (extent[2] - extent[0]) / width
    const y_rs = (extent[3] - extent[1]) / height
    const resolution = Math.max(x_rs, y_rs)
    const closest = this._resolutions.reduce(function(previous, current) {
      if (Math.abs(current - resolution) < Math.abs(previous - resolution))
        return current
      else
        return previous
    })
    return this._resolutions.indexOf(closest)
  }

  snap_to_zoom_level(extent: Extent, height: number, width: number, level: number): Extent {
    const [xmin, ymin, xmax, ymax] = extent
    const desired_res = this._resolutions[level]
    let desired_x_delta = width * desired_res
    let desired_y_delta = height * desired_res
    if (!this.snap_to_zoom) {
      const xscale = (xmax-xmin)/desired_x_delta
      const yscale = (ymax-ymin)/desired_y_delta
      if (xscale > yscale) {
        desired_x_delta = (xmax-xmin)
        desired_y_delta = desired_y_delta*xscale
      } else {
        desired_x_delta = desired_x_delta*yscale
        desired_y_delta = (ymax-ymin)
      }
    }
    const x_adjust = (desired_x_delta - (xmax - xmin)) / 2
    const y_adjust = (desired_y_delta - (ymax - ymin)) / 2

    return [xmin - x_adjust, ymin - y_adjust, xmax + x_adjust, ymax + y_adjust]
  }

  tms_to_wmts(x: number, y: number, z: number): [number, number, number] {
    // Note this works both ways
    return [x, 2**z - 1 - y, z]
  }

  wmts_to_tms(x: number, y: number, z: number): [number, number, number] {
    // Note this works both ways
    return [x, 2**z - 1 - y, z]
  }

  pixels_to_meters(px: number, py: number, level: number): [number, number] {
    const res = this.get_resolution(level)
    const mx = (px * res) - this.x_origin_offset
    const my = (py * res) - this.y_origin_offset
    return [mx, my]
  }

  meters_to_pixels(mx: number, my: number, level: number): [number, number] {
    const res = this.get_resolution(level)
    const px = (mx + this.x_origin_offset) / res
    const py = (my + this.y_origin_offset) / res
    return [px, py]
  }

  pixels_to_tile(px: number, py: number): [number, number] {
    let tx = Math.ceil(px / this.tile_size)
    tx = tx === 0 ? tx : tx - 1
    const ty = Math.max(Math.ceil(py / this.tile_size) - 1, 0)
    return [tx, ty]
  }

  pixels_to_raster(px: number, py: number, level: number): [number, number] {
    const mapSize = this.tile_size << level
    return [px, mapSize - py]
  }

  meters_to_tile(mx: number, my: number, level: number): [number, number] {
    const [px, py] = this.meters_to_pixels(mx, my, level)
    return this.pixels_to_tile(px, py)
  }

  get_tile_meter_bounds(tx: number, ty: number, level: number): Bounds {
    // expects tms styles coordinates (bottom-left origin)
    const [xmin, ymin] = this.pixels_to_meters(tx * this.tile_size, ty * this.tile_size, level)
    const [xmax, ymax] = this.pixels_to_meters((tx + 1) * this.tile_size, (ty + 1) * this.tile_size, level)
    return [xmin, ymin, xmax, ymax]
  }

  get_tile_geographic_bounds(tx: number, ty: number, level: number): Bounds {
    const bounds = this.get_tile_meter_bounds(tx, ty, level)
    const [minLon, minLat, maxLon, maxLat] = meters_extent_to_geographic(bounds)
    return [minLon, minLat, maxLon, maxLat]
  }

  get_tiles_by_extent(extent: Extent, level: number, tile_border: number = 1): [number, number, number, Bounds][] {
    // unpack extent and convert to tile coordinates
    const [xmin, ymin, xmax, ymax] = extent
    let [txmin, tymin] = this.meters_to_tile(xmin, ymin, level)
    let [txmax, tymax] = this.meters_to_tile(xmax, ymax, level)

    // add tiles which border
    txmin -= tile_border
    tymin -= tile_border
    txmax += tile_border
    tymax += tile_border

    const tiles: [number, number, number, Bounds][] = []
    for (let ty = tymax; ty >= tymin; ty--) {
      for (let tx = txmin; tx <= txmax; tx++) {
        if (this.is_valid_tile(tx, ty, level))
          tiles.push([tx, ty, level, this.get_tile_meter_bounds(tx, ty, level)])
      }
    }

    this.sort_tiles_from_center(tiles, [txmin, tymin, txmax, tymax])
    return tiles
  }

  quadkey_to_tile_xyz(quadKey: string): [number, number, number] {
    /**
     * Computes tile x, y and z values based on quadKey.
     */
    let tileX = 0
    let tileY = 0
    const tileZ = quadKey.length
    for (let i = tileZ; i > 0; i--) {
      const value = quadKey.charAt(tileZ - i)
      const mask = 1 << (i - 1)

      switch (value) {
        case "0":
          continue
        case "1":
          tileX |= mask
          break
        case "2":
          tileY |= mask
          break
        case "3":
          tileX |= mask
          tileY |= mask
          break
        default:
          throw new TypeError(`Invalid Quadkey: ${quadKey}`)
      }
    }

    return [tileX, tileY, tileZ]
  }

  tile_xyz_to_quadkey(x: number, y: number, z: number): string {
    /*
     * Computes quadkey value based on tile x, y and z values.
     */
    let quadkey = ""
    for (let i = z; i > 0; i--) {
      const mask = 1 << (i - 1)
      let digit = 0
      if ((x & mask) !== 0) {
        digit += 1
      }
      if ((y & mask) !== 0) {
        digit += 2
      }
      quadkey += digit.toString()
    }
    return quadkey
  }

  children_by_tile_xyz(x: number, y: number, z: number): [number, number, number, Bounds][] {
    const quadkey = this.tile_xyz_to_quadkey(x, y, z)
    const child_tile_xyz: [number, number, number, Bounds][] = []

    for (let i = 0; i <= 3; i++) {
      const [x, y, z] = this.quadkey_to_tile_xyz(quadkey + i.toString())
      const b = this.get_tile_meter_bounds(x, y, z)
      child_tile_xyz.push([x, y, z, b])
    }

    return child_tile_xyz
  }

  get_closest_parent_by_tile_xyz(x: number, y: number, z: number): [number, number, number] {
    const world_x = this.calculate_world_x_by_tile_xyz(x, y, z)
    ;[x, y, z] = this.normalize_xyz(x, y, z)
    let quadkey = this.tile_xyz_to_quadkey(x, y, z)
    while (quadkey.length > 0) {
      quadkey = quadkey.substring(0, quadkey.length - 1)
      ;[x, y, z] = this.quadkey_to_tile_xyz(quadkey)
      ;[x, y, z] = this.denormalize_xyz(x, y, z, world_x)
      if (this.tiles.has(this.tile_xyz_to_key(x, y, z)))
        return [x, y, z]
    }
    return [0, 0, 0]
  }

  normalize_xyz(x: number, y: number, z: number): [number, number, number] {
    if (this.wrap_around) {
      const tile_count = 2**z
      return [((x % tile_count) + tile_count) % tile_count, y, z]
    } else {
      return [x, y, z]
    }
  }

  denormalize_xyz(x: number, y: number, z: number, world_x: number): [number, number, number] {
    return [x + (world_x * 2**z), y, z]
  }

  denormalize_meters(meters_x: number, meters_y: number, _level: number, world_x: number): [number, number] {
    return [meters_x + (world_x * 2 * Math.PI * 6378137), meters_y]
  }

  calculate_world_x_by_tile_xyz(x: number, _y: number, z: number): number {
    return Math.floor(x / 2**z)
  }
}
'''

class my_TileSource(TileSource):
    ''' A base class for Mercator tile services (e.g. ``WMTSTileSource``).

    '''

    #__implementation__ = TypeScript(TS_CODE)

    _args = ('url', 'tile_size', 'min_zoom', 'max_zoom', 'x_origin_offset', 'y_origin_offset', 'extra_url_vars', 'initial_resolution')

    x_origin_offset = Override(default=0)

    y_origin_offset = Override(default=0)

    initial_resolution = Override(default=15654)

    tile_size = Override(default=254)

class MyMercatorTileSource(TileSource):
    ''' A base class for Mercator tile services (e.g. ``WMTSTileSource``).

    '''

    #__implementation__ = 'C:/Users/meast/anaconda3/envs/bkdev/Lib/site-packages/bokeh-3.0.0.dev1+47.g281e3f87d-py3.9.egg/bokeh/server/static/js/lib/models/tiles/my_mercator_tile_source.js'

    _args = ('url', 'tile_size', 'min_zoom', 'max_zoom', 'x_origin_offset', 'y_origin_offset', 'extra_url_vars', 'initial_resolution')

    x_origin_offset = Override(default=0)

    y_origin_offset = Override(default=0)

    initial_resolution = Override(default=15654)

    tile_size= Override(default=254)

    snap_to_zoom = Bool(default=False, help="""
    Forces initial extents to snap to the closest larger zoom level.""")

    wrap_around = Bool(default=False, help="""
    Enables continuous horizontal panning by wrapping the x-axis based on
    bounds of map.

    ..note::
        Axis coordinates are not wrapped. To toggle axis label visibility,
        use ``plot.axis.visible = False``.

    """)

slide_width=180848
tile_size=256
init_res=slide_width/tile_size
mpp=0.5015

def invert_y(data):
  feats=[]
  for feat in range(len(data['features'])):
    if data['features'][feat]['geometry']['type']=='MultiPolygon':
      #skip.append(feat)
      continue
    for i in range(len(data['features'][feat]['geometry']['coordinates'][0])):
      data['features'][feat]['geometry']['coordinates'][0][i][1]=-data['features'][feat]['geometry']['coordinates'][0][i][1]
    if feat%10==0:
      feats.append(data['features'][feat])
  data['features']=feats

sf=1
#wsi_provider=TileProvider(name="WSI provider", url=r'http://127.0.0.1:5000/slide_files/{z}/{x}_{y}.jpeg', attribution="", size=254)
def make_ts(route):
  sf=1
  ts=WMTSTileSource(name="WSI provider", url=route, attribution="")
  ts.tile_size=256
  ts.initial_resolution=40211.5*sf*(2/(100*pi))   #156543.03392804097    40030 great circ
  ts.x_origin_offset=0#5000000
  #ts.y_origin_offset=-2500000
  ts.y_origin_offset=10247680*sf*(2/(100*pi))  + 438.715 +38.997+13-195.728  #10160000,   509.3
  ts.wrap_around=False
  ts.max_zoom=9
  #ts.min_zoom=10
  return ts

#ts1=make_ts(r'http://127.0.0.1:5000/slide_files/{z}/{x}_{y}.jpeg')
ts1=make_ts(r'http://127.0.0.1:5000/layer/slide/zoomify/TileGroup1/{z}-{x}-{y}.png')
ts2=make_ts(r'http://127.0.0.1:5000/layer/mask/zoomify/TileGroup1/{z}-{x}-{y}.png')


base_path=Path(r'E:\Meso_TCGA\slides_tiled\TCGA-SC-A6LN-01Z-00-DX1.379BF588-5A65-4BF8-84CF-5136085D8A47\graph.npz')

with np.load(base_path) as dat:
  X=dat['X']
  c=dat['c']
  it=dat['it']

X=X[0:-1:10,:]
c=c[0:-1:10,:]
it=it[0:-1:10,0]
cvals=0.5+(c[:,0]**2-c[:,1]**2)/2
cvals[it==False]=0
coord_sf=1#155.027

source = ColumnDataSource(data=dict(x=coord_sf*X[:,0], y=-coord_sf*X[:,1], c=cvals,))

TOOLTIPS=[
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("Scores", "[@c1, @c2]"),
      ]

objpath=Path(r'E:\Meso_TCGA\A6LN_example2.geojson')
#objpath=Path(r'E:\Meso_TCGA\A6LN_allcells_nofeats.geojson')
with open(objpath,'rb') as gf:
  data = json.load(gf)
invert_y(data)
patch_source = GeoJSONDataSource(geojson=json.dumps(data))

#p = figure(x_range=(0, 47194), y_range=(0, 180848),x_axis_type="linear", y_axis_type="linear")
#p = figure(x_range=(0, 180848), y_range=(0,47194),x_axis_type="mercator", y_axis_type="mercator", width=1200,height=800)
p = figure(x_range=(0, 70000), y_range=(0,-50000),x_axis_type="linear", y_axis_type="linear",
   width=1500,height=900, tooltips=TOOLTIPS, lod_factor=20,output_backend="webgl")
p.add_tile(ts1)
p.add_tile(ts2)
p.grid.grid_line_color=None

circ_cmapper=linear_cmap(field_name='c', palette='RdYlGn11' ,low=0.001 ,high=1, high_color=(255,255,255,1) ,low_color=(255,255,255,1))
#nodes = p.circle('x', 'y', size=10, color=circ_cmapper, alpha=1.0, source = source)
#pat = p.patches('xs','ys', source=patch_source, fill_alpha=0.1, line_alpha=1.0)
nodes=None
p.renderers[0].tile_source.max_zoom=8
p.renderers[0].tile_source.max_zoom=8
#p.renderers[0].tile_source.initial_resolution=4000
p.renderers[0].smoothing=False
#p.renderers[1].smoothing=False

slider1 = Slider(
        title="Adjust alpha WSI",
        start=0,
        end=1,
        step=0.05,
        value=1.0
    )

slider2 = Slider(
        title="Adjust alpha Mask",
        start=0,
        end=1,
        step=0.05,
        value=0.8
    )

toggle1 = Toggle(label="Show Dots", button_type="success")
toggle2 = Toggle(label="Show Mask", button_type="success")

callback1 = CustomJS(args=dict(n=nodes, s=slider1), code="""
        if (this.active) {
            n.visible = false;
        } else {
            n.glyph.line_alpha = s.value;
            n.visible = true;
        };
    """)

callback2=CustomJS(args=dict(p=p,s=slider2), code="""
        if (p.renderers[1].alpha==0) {
        p.renderers[1].alpha=s.value;
        }
        else {
          p.renderers[1].alpha=0.0
        }

    """)

toggle1.js_on_click(callback1)
toggle2.js_on_click(callback2)

slidercb1=CustomJS(args=dict(p=p,s=slider1), code="""
        p.renderers[0].alpha=s.value;
        p.renderers[0].tile_source.max_zoom=7
        p.renderers[1].tile_source.max_zoom=7

    """)

slidercb2=CustomJS(args=dict(p=p,s=slider2), code="""
        p.renderers[1].alpha=s.value;

    """)

slider1.js_on_change('value', slidercb1)
slider2.js_on_change('value', slidercb2)


gr = layout(
        [
            [p, [slider1,slider2, toggle1, toggle2]],#, toggle1, toggle2, toggle3, drop_i, drop_f, div]],
        ]
    )

show(gr)
