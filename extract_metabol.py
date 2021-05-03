# this script will download and assemble the metabolic pathways poster from biochemical-pathways.com
# Usnish Majumdar, 10/21/16

# check python version before urllib
import itertools
import sys

if sys.version_info[0] < 3:
    raise Exception("Python Version > 3 is required.")

import urllib.request
import subprocess
from PIL import Image
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, List
from tqdm import tqdm
import numpy as np


@dataclass(frozen=True)
class ZoomLevelDefinition:
    id: int
    columns: int
    rows: int

    tile_width: int = 1024
    tile_height: int = 1024

    @property
    def total_width(self):
        return self.columns * self.tile_width

    @property
    def total_height(self):
        return self.rows * self.tile_height


ZOOM_LEVELS = {
    0: ZoomLevelDefinition(0, columns=1, rows=1),
    1: ZoomLevelDefinition(1, columns=2, rows=2),
    2: ZoomLevelDefinition(2, columns=4, rows=3),
    3: ZoomLevelDefinition(3, columns=7, rows=5),
    4: ZoomLevelDefinition(4, columns=14, rows=10),
    5: ZoomLevelDefinition(5, columns=27, rows=19),
    6: ZoomLevelDefinition(6, columns=54, rows=38),
}

# these are the eight layers of images that come together on the website,
# in order of z-position from back to front
LAYERS = ['schematicOverview', 'grid', 'background', 'enzymes', 'coenzymes', 'substrates',
          'regulatoryEffects', 'higherPlants', 'unicellularOrganisms']


def download_tile(zoom_level: ZoomLevelDefinition,
                  layer: str,
                  row: int,
                  col: int,
                  destination_dir: Path) -> Path:
    # noinspection HttpUrlsUsage
    url = f'http://mapserver1.biochemical-pathways.com/map1/{layer}/{zoom_level.id}/{col}/{row}.png?v=4'
    filename = destination_dir / f'{layer}_{zoom_level.id}_{row:02}_{col:02}.png'
    urllib.request.urlretrieve(url, filename)

    # at higher magnification, some of the grid images are actually blank 1x1 px
    # images. We have to scale them back so that they'll tile properly
    with Image.open(filename) as im:
        if im.size[0] < zoom_level.tile_width:
            im.resize((zoom_level.tile_width, zoom_level.tile_height)).save(filename)

    return filename


def download_layer(zoom_level: ZoomLevelDefinition, layer: str):
    tiles_dir = Path('images')
    tiles_dir.mkdir(exist_ok=True)
    for row, col in tqdm(list(itertools.product(range(zoom_level.rows), range(zoom_level.columns))),
                         desc=f'Downloading tiles for layer "{layer}"', unit='tile'):
        download_tile(zoom_level, layer, row, col, tiles_dir)


def assemble_tiles(zoom_level: ZoomLevelDefinition, layer: str):
    tiles_dir = Path('images')
    assembled_dir = Path('assembled')
    assembled_dir.mkdir(exist_ok=True)

    tile_size = f'{zoom_level.columns}x{zoom_level.rows}'
    filename = tiles_dir / f'{layer}_{zoom_level.id}_*.png'
    outfile = assembled_dir / f'{layer}_{zoom_level.id}.png'
    command = f'montage -mode concatenate -background none -tile {tile_size} {filename} {outfile}'

    subprocess.call(command, shell=True)


def determine_symmetric_crop_size(zoom_level: ZoomLevelDefinition) -> Tuple[int, int]:
    # row -> column, column -> row
    def other_axis(this_axis):
        return 1 if this_axis == 0 else 0

    # index of first non-zero value along axis
    def first_non_zero(a, axis) -> int:
        return (a != 0).any(axis=other_axis(axis)).argmax()

    # index of last non-zero value along axis
    def last_non_zero(a, axis) -> int:
        return a.shape[axis] - first_non_zero(a[::-1, ::-1], axis) - 1

    tiles_dir = Path('images')
    tiles_dir.mkdir(exist_ok=True)
    with Image.open(download_tile(zoom_level, 'grid',
                                  0, 0,
                                  tiles_dir)) as top_left, \
            Image.open(download_tile(zoom_level, 'grid',
                                     zoom_level.rows - 1, zoom_level.columns - 1,
                                     tiles_dir)) as bottom_right:
        top_left_alpha = np.array(top_left)[..., 3]
        bottom_right_alpha = np.array(bottom_right)[..., 3]

        top_padding = first_non_zero(top_left_alpha, axis=0)
        left_padding = first_non_zero(top_left_alpha, axis=1)

        bottom_padding = bottom_right_alpha.shape[0] - last_non_zero(bottom_right_alpha, 0) - 1
        right_padding = bottom_right_alpha.shape[1] - last_non_zero(bottom_right_alpha, 1) - 1

        return (right_padding - left_padding,
                bottom_padding - top_padding)


def reorder_layers(layers: List[str],
                   layer_name: str,
                   include: bool,
                   foreground: bool) -> List[str]:
    if not include or foreground:
        layers = [x
                  for x in layers
                  if x != layer_name]
        if foreground:
            layers += [layer_name]
    return layers


def multiply_opacity(image: Image, factor: float) -> Image:
    image = image.copy().convert('RGBA')
    alpha = np.array(image)[..., -1]
    image.putalpha(Image.fromarray((alpha * factor).astype('uint8')))
    return image


def main(zoom_level: ZoomLevelDefinition,
         destination: Union[Path, str],
         include_grid: bool = False,
         grid_in_foreground: bool = False,
         schematic_overview_opacity: float = 0,
         schematic_overview_in_foreground: bool = False):
    # start with a blank canvas
    canvas = Image.new('RGB', (zoom_level.total_width, zoom_level.total_height), 'white')

    in_progress_dir = Path('in-progress')
    in_progress_dir.mkdir(exist_ok=True)
    assembled_dir = Path('assembled')

    included_layers = LAYERS
    included_layers = reorder_layers(included_layers,
                                     'grid',
                                     include=include_grid,
                                     foreground=grid_in_foreground)
    included_layers = reorder_layers(included_layers,
                                     'schematicOverview',
                                     include=schematic_overview_opacity != 0,
                                     foreground=schematic_overview_in_foreground)

    for layer in tqdm(included_layers, desc='Layers', unit='layer'):
        download_layer(zoom_level, layer)
        assemble_tiles(zoom_level, layer)

        with Image.open(assembled_dir / f'{layer}_{zoom_level.id}.png') as layer_image:
            layer_image = layer_image.convert('RGBA')

            if layer == 'schematicOverview' and schematic_overview_opacity != 1:
                layer_image = multiply_opacity(layer_image, schematic_overview_opacity)

            canvas.paste(layer_image, mask=layer_image)
            canvas.save(in_progress_dir / f'progress_{layer}.png', mode='RGBA')

    crop_size = determine_symmetric_crop_size(zoom_level)
    canvas = canvas.crop((0, 0, canvas.width - crop_size[0], canvas.height - crop_size[1]))
    canvas.save(destination)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--zoom', choices=range(7), default=3, metavar='ZOOM', type=int,
                        help='The zoom level if the downloaded tiles, higher ZOOM results in higher resolution')
    parser.add_argument('--include-grid', default=False, action='store_true',
                        help='The grid layer is excluded by default. Use this to re-enable it')
    parser.add_argument('--grid-in-foreground', default=False, action='store_true',
                        help='This script displays the grid as the bottom layer by default. '
                             'The online interface displays the grid as the foremost layer, hiding contents behind it. '
                             'Use this flag to bring back the online behaviour. '
                             'Requires --include-grid in order to have an effect.')
    parser.add_argument('--schematic-overview-opacity', default=0, metavar='OPACITY', type=float,
                        help='Include the schematic overview layer as a background layer with opacity OPACITY. '
                             'The layer is not included by default.')
    parser.add_argument('--schematic-overview-in-foreground', default=False, action='store_true',
                        help='This script displays the schematic overview layer in the background by default. '
                             'The online interface displays the layer as the foremost layer, '
                             'hiding pathways behind it. Use this flag to bring back the online behaviour. '
                             'Requires --schematic-overview-opacity to have a value greater than 0 '
                             'in order to have an effect.')
    parser.add_argument('destination', type=Path, default=Path('finalimg.png'), nargs='?', metavar='PATH',
                        help='Where to save the final image')

    args = parser.parse_args()

    main(zoom_level=ZOOM_LEVELS[args.zoom],
         include_grid=args.include_grid,
         destination=args.destination,
         grid_in_foreground=args.grid_in_foreground,
         schematic_overview_opacity=args.schematic_overview_opacity,
         schematic_overview_in_foreground=args.schematic_overview_in_foreground)
