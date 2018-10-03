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


class HDFEOSFileReader(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(HDFEOSFileReader, self).__init__(filename, filename_info, filetype_info)
        try:
            self.sd = SD(str(self.filename))
        except HDF4Error as err:
            raise ValueError("Could not load data from " + str(self.filename)
                             + ": " + str(err))
        self.metadata = self.read_mda(self.sd.attributes()['CoreMetadata.0'])
        self.metadata.update(self.read_mda(
            self.sd.attributes()['StructMetadata.0']))
        self.metadata.update(self.read_mda(
            self.sd.attributes()['ArchiveMetadata.0']))

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


class HDFEOSGeoReader(HDFEOSFileReader):

    def __init__(self, filename, filename_info, filetype_info):
        HDFEOSFileReader.__init__(self, filename, filename_info, filetype_info)

        ds = self.metadata['INVENTORYMETADATA'][
            'COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE']
        if ds.endswith('D03'):
            self.resolution = 1000
        else:
            self.resolution = 5000
        self.cache = {}
        self.cache[250] = {}
        self.cache[250]['lons'] = None
        self.cache[250]['lats'] = None

        self.cache[500] = {}
        self.cache[500]['lons'] = None
        self.cache[500]['lats'] = None

        self.cache[1000] = {}
        self.cache[1000]['lons'] = None
        self.cache[1000]['lats'] = None

    def get_dataset(self, key, info, out=None, xslice=None, yslice=None):
        """Get the dataset designated by *key*."""
        if key.name in ['solar_zenith_angle', 'solar_azimuth_angle',
                        'satellite_zenith_angle', 'satellite_azimuth_angle']:

            if key.name == 'solar_zenith_angle':
                var = self.sd.select('SolarZenith')
            if key.name == 'solar_azimuth_angle':
                var = self.sd.select('SolarAzimuth')
            if key.name == 'satellite_zenith_angle':
                var = self.sd.select('SensorZenith')
            if key.name == 'satellite_azimuth_angle':
                var = self.sd.select('SensorAzimuth')

            data = xr.DataArray(from_sds(var, chunks=CHUNK_SIZE),
                                dims=['y', 'x']).astype(np.float32)
            data = data.where(data != var._FillValue)
            data = data * np.float32(var.scale_factor)

            data.attrs = info
            return data

        if key.name not in ['longitude', 'latitude']:
            return

        if (self.cache[key.resolution]['lons'] is None or
                self.cache[key.resolution]['lats'] is None):

            lons_id = DatasetID('longitude',
                                resolution=key.resolution)
            lats_id = DatasetID('latitude',
                                resolution=key.resolution)

            lons, lats = self.load(
                [lons_id, lats_id], interpolate=False, raw=True)
            if key.resolution != self.resolution:
                from geotiepoints.geointerpolator import GeoInterpolator
                lons, lats = self._interpolate([lons, lats],
                                               self.resolution,
                                               lons_id.resolution,
                                               GeoInterpolator)
                lons = np.ma.masked_invalid(np.ascontiguousarray(lons))
                lats = np.ma.masked_invalid(np.ascontiguousarray(lats))
            self.cache[key.resolution]['lons'] = lons
            self.cache[key.resolution]['lats'] = lats

        if key.name == 'latitude':
            data = self.cache[key.resolution]['lats'].filled(np.nan)
            data = xr.DataArray(da.from_array(data, chunks=(CHUNK_SIZE, CHUNK_SIZE)),
                                dims=['y', 'x'])
        else:
            data = self.cache[key.resolution]['lons'].filled(np.nan)
            data = xr.DataArray(da.from_array(data, chunks=(CHUNK_SIZE,
                                                            CHUNK_SIZE)),
                                dims=['y', 'x'])
        data.attrs = info
        return data

    def load(self, keys, interpolate=True, raw=False):
        """Load the data."""
        projectables = []
        for key in keys:
            dataset = self.sd.select(key.name.capitalize())
            fill_value = dataset.attributes()["_FillValue"]
            try:
                scale_factor = dataset.attributes()["scale_factor"]
            except KeyError:
                scale_factor = 1
            data = np.ma.masked_equal(dataset.get(), fill_value) * scale_factor

            # TODO: interpolate if needed
            if (key.resolution is not None and
                    key.resolution < self.resolution and
                    interpolate):
                data = self._interpolate(data, self.resolution, key.resolution)
            if not raw:
                data = data.filled(np.nan)
                data = xr.DataArray(da.from_array(data, chunks=(CHUNK_SIZE,
                                                                CHUNK_SIZE)),
                                    dims=['y', 'x'])
            projectables.append(data)

        return projectables

    @staticmethod
    def _interpolate(data, coarse_resolution, resolution, interpolator=None):
        if resolution == coarse_resolution:
            return data

        if interpolator is None:
            from geotiepoints.interpolator import Interpolator
            interpolator = Interpolator

        logger.debug("Interpolating from " + str(coarse_resolution)
                     + " to " + str(resolution))

        if isinstance(data, (tuple, list, set)):
            lines = data[0].shape[0]
        else:
            lines = data.shape[0]

        if coarse_resolution == 5000:
            coarse_cols = np.arange(2, 1354, 5)
            lines *= 5
            coarse_rows = np.arange(2, lines, 5)

        elif coarse_resolution == 1000:
            coarse_cols = np.arange(1354)
            coarse_rows = np.arange(lines)

        if resolution == 1000:
            fine_cols = np.arange(1354)
            fine_rows = np.arange(lines)
            chunk_size = 10
        elif resolution == 500:
            fine_cols = np.arange(1354 * 2) / 2.0
            fine_rows = (np.arange(lines * 2) - 0.5) / 2.0
            chunk_size = 20
        elif resolution == 250:
            fine_cols = np.arange(1354 * 4) / 4.0
            fine_rows = (np.arange(lines * 4) - 1.5) / 4.0
            chunk_size = 40

        along_track_order = 1
        cross_track_order = 3

        satint = interpolator(data,
                              (coarse_rows, coarse_cols),
                              (fine_rows, fine_cols),
                              along_track_order,
                              cross_track_order,
                              chunk_size=chunk_size)

        satint.fill_borders("y", "x")
        return satint.interpolate()


class HDFEOSBandReader(HDFEOSFileReader):

    def __init__(self, filename, filename_info, filetype_info):
        HDFEOSFileReader.__init__(self, filename, filename_info, filetype_info)

        ds = self.metadata['INVENTORYMETADATA'][
            'COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE']

    def get_dataset(self, key, info):

        print(key, info)

        """Read data from file and return the corresponding projectables."""
        datadict = {
            1000: ['EV_250_Aggr1km_RefSB',
                   'EV_500_Aggr1km_RefSB',
                   'EV_1KM_RefSB',
                   'EV_1KM_Emissive'],
            500: ['EV_250_Aggr500_RefSB',
                  'EV_500_RefSB'],
            250: ['EV_250_RefSB']}

        platform_name = self.metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR'][
            'ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDPLATFORMSHORTNAME']['VALUE']

        info.update({'platform_name': 'EOS-' + platform_name})
        info.update({'sensor': 'modis'})

        datasets = datadict[self.resolution]
        for dataset in datasets:
            subdata = self.sd.select(dataset)
            var_attrs = subdata.attributes()
            band_names = var_attrs["band_names"].split(",")

            # get the relative indices of the desired channel
            try:
                index = band_names.index(key.name)
            except ValueError:
                continue
            uncertainty = self.sd.select(dataset + "_Uncert_Indexes")
            array = xr.DataArray(from_sds(subdata, chunks=CHUNK_SIZE)[index, :, :],
                                 dims=['y', 'x']).astype(np.float32)
            valid_range = var_attrs['valid_range']
            array = array.where(array >= np.float32(valid_range[0]))
            array = array.where(array <= np.float32(valid_range[1]))
            array = array.where(from_sds(uncertainty, chunks=CHUNK_SIZE)[index, :, :] < 15)

            if key.calibration == 'brightness_temperature':
                projectable = calibrate_bt(array, var_attrs, index, key.name)
                info.setdefault('units', 'K')
                info.setdefault('standard_name', 'toa_brightness_temperature')
            elif key.calibration == 'reflectance':
                projectable = calibrate_refl(array, var_attrs, index)
                info.setdefault('units', '%')
                info.setdefault('standard_name',
                                'toa_bidirectional_reflectance')
            elif key.calibration == 'radiance':
                projectable = calibrate_radiance(array, var_attrs, index)
                info.setdefault('units', var_attrs.get('radiance_units'))
                info.setdefault('standard_name',
                                'toa_outgoing_radiance_per_unit_wavelength')
            elif key.calibration == 'counts':
                projectable = calibrate_counts(array, var_attrs, index)
                info.setdefault('units', 'counts')
                info.setdefault('standard_name', 'counts')  # made up
            else:
                raise ValueError("Unknown calibration for "
                                 "key: {}".format(key))
            projectable.attrs = info

            return projectable

    # These have to be interpolated...
    def get_height(self):
        return self.data.select("Height")

    def get_sunz(self):
        return self.data.select("SolarZenith")

    def get_suna(self):
        return self.data.select("SolarAzimuth")

    def get_satz(self):
        return self.data.select("SensorZenith")

    def get_sata(self):
        return self.data.select("SensorAzimuth")
