#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2014, 2017.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Ronald Scheirer <ronald.scheirer@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to Modis level 1b format.
http://www.icare.univ-lille1.fr/wiki/index.php/MODIS_geolocation
http://www.sciencedirect.com/science?_ob=MiamiImageURL&_imagekey=B6V6V-4700BJP-\
3-27&_cdi=5824&_user=671124&_check=y&_orig=search&_coverDate=11%2F30%2F2002&vie\
w=c&wchp=dGLzVlz-zSkWz&md5=bac5bc7a4f08007722ae793954f1dd63&ie=/sdarticle.pdf
"""

import logging
from datetime import datetime

import numpy as np
from pyhdf.error import HDF4Error
from pyhdf.SD import SD

import dask.array as da
import xarray.ufuncs as xu
import xarray as xr
from satpy import CHUNK_SIZE
from satpy.dataset import DatasetID
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hdf4_utils import from_sds

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'O': 'MODIS-Terra',
                  'Y': 'MODIS-Aqua'}

class HDFEOSBandReader_L2(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):

        super(HDFEOSBandReader_L2, self).__init__(filename, filename_info,
                                         filetype_info)

        chunks = {'1km Data Lines:MODIS SWATH TYPE L2': CHUNK_SIZE,
                  '1km Data Samples:MODIS SWATH TYPE L2': CHUNK_SIZE,
                  '250m Data Lines:MODIS SWATH TYPE L2': CHUNK_SIZE,
                  '250m Data Samples:MODIS SWATH TYPE L2': CHUNK_SIZE,
                  '500m Data Lines:MODIS SWATH TYPE L2': CHUNK_SIZE,
                  '500m Data Samples:MODIS SWATH TYPE L2': CHUNK_SIZE}

        self.nc = xr.open_dataset(filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks=chunks)

        # rename dimensions to x/y
        # anything with Samples -> x
        # anything with Lines   -> y
        dim_mapper = {}

        for dim_name in self.nc.dims:
            if "Samples" in dim_name:
                dim_mapper[dim_name] = 'x'
            elif "Lines" in dim_name:
                dim_mapper[dim_name] = 'y'
            else:
                logger.debug('Unexpected dimension name: %s', dim_name)

        self.nc = self.nc.rename(dim_mapper)

        # TODO: get metadata from the manifest file (xfdumanifest.xml)
        self.platform_name = PLATFORM_NAMES[filename_info['platform_indicator']]
        self.sensor = 'modis'

        self.metadata = self.read_mda(self.nc.attrs['CoreMetadata.0'])
        self.metadata.update(self.read_mda(self.nc.attrs['StructMetadata.0']))
        self.metadata.update(self.read_mda(self.nc.attrs['ArchiveMetadata.0']))

    @property
    def start_time(self):
        date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGDATE']['VALUE'] + ' ' +
                self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGTIME']['VALUE'])
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

    @property
    def end_time(self):
        date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGDATE']['VALUE'] + ' ' +
                self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGTIME']['VALUE'])
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

    def get_dataset(self, key, info):
        """Load a dataset."""

        logger.debug('Reading %s.', key.name)

        band_name, resolution = info['name'].split('_')
        band_num = band_name[1:]

        ds_name = '{resolution} Surface Reflectance Band {band_num}'.format(resolution=resolution, 
                                                                            band_num=band_num)

        ds = self.nc[ds_name]

        # print(ds.attrs)

        scale_factor = 1 / ds.attrs['scale_factor']
        scale_offset = ds.attrs['add_offset']

        ds = ds.where(ds != ds.attrs['_FillValue'])
        ds = ds * scale_factor + scale_offset

        return ds

    def read_mda(self, attribute):

        raw_lines = attribute.split('\n')
        raw_lines = list(filter(lambda x: len(x)>0, raw_lines))

        lines = []

        for line in raw_lines:

            line = (5*' ').join(line.split('\t'))
            
            if line.strip() == 'END':
                lines.append(line)
            elif '=' not in line:
                # print(line)
                lines[-1] += line.strip()
            else:
                lines.append(line)

        mda = {}
        current_dict = mda
        path = []

        for line in lines:

            if not line:
                continue
            if line == 'END':
                break

            if '=' not in line:
                continue

            key, val = line.split('=', maxsplit=1)

            if isinstance(val, list):
                val = '='.join(val)

            key = key.strip()
            val = val.strip()

            try:
                val = eval(val)
            except NameError:
                pass

            if key in ['GROUP', 'OBJECT']:
                new_dict = {}
                path.append(val)
                current_dict[val] = new_dict
                current_dict = new_dict
            elif key in ['END_GROUP', 'END_OBJECT']:
                if val != path[-1]:
                    raise SyntaxError
                path = path[:-1]
                current_dict = mda
                for item in path:
                    current_dict = current_dict[item]
            elif key in ['CLASS', 'NUM_VAL']:
                pass
            else:
                current_dict[key] = val

        return mda