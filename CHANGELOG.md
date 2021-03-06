###############################################################################
## Version 0.9.0 (2018/07/02)

### Issues Closed

* [Issue 344](https://github.com/pytroll/satpy/issues/344) - find_files_and_reader does not seem to care about start_time! ([PR 349](https://github.com/pytroll/satpy/pull/349))
* [Issue 338](https://github.com/pytroll/satpy/issues/338) - Creating a Scene object with Terra MODIS data
* [Issue 332](https://github.com/pytroll/satpy/issues/332) - Non-requested datasets are saved when composites fail to generate ([PR 342](https://github.com/pytroll/satpy/pull/342))

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 355](https://github.com/pytroll/satpy/pull/355) - Fix ABI L1B reader losing file variable attributes
* [PR 353](https://github.com/pytroll/satpy/pull/353) - Fix multiscene memory issues by adding an optional batch_size
* [PR 351](https://github.com/pytroll/satpy/pull/351) - Fix AMSR-2 L1B reader loading bytes incorrectly
* [PR 349](https://github.com/pytroll/satpy/pull/349) - Fix datetime-based file selection when filename only has a start time ([344](https://github.com/pytroll/satpy/issues/344))
* [PR 348](https://github.com/pytroll/satpy/pull/348) - Fix freezing of areas before resampling even as strings
* [PR 343](https://github.com/pytroll/satpy/pull/343) - Fix shape assertion after resampling
* [PR 342](https://github.com/pytroll/satpy/pull/342) - Fix Scene save_datasets to only save datasets from the wishlist ([332](https://github.com/pytroll/satpy/issues/332))
* [PR 341](https://github.com/pytroll/satpy/pull/341) - Fix ancillary variable loading when anc var is already loaded
* [PR 340](https://github.com/pytroll/satpy/pull/340) - Cut radiances array depending on number of scans

In this release 9 pull requests were closed.


## Version 0.9.0b0 (2018/06/26)

### Issues Closed

* [Issue 328](https://github.com/pytroll/satpy/issues/328) - hrit reader bugs ([PR 329](https://github.com/pytroll/satpy/pull/329))
* [Issue 323](https://github.com/pytroll/satpy/issues/323) - "Manual" application of corrections
* [Issue 320](https://github.com/pytroll/satpy/issues/320) - Overview of code layout
* [Issue 279](https://github.com/pytroll/satpy/issues/279) - Add 'level' to DatasetID ([PR 283](https://github.com/pytroll/satpy/pull/283))
* [Issue 272](https://github.com/pytroll/satpy/issues/272) - How to save region of interest from Band 3 Himawari Data as png image ([PR 276](https://github.com/pytroll/satpy/pull/276))
* [Issue 267](https://github.com/pytroll/satpy/issues/267) - Missing dependency causes strange error during unit tests ([PR 273](https://github.com/pytroll/satpy/pull/273))
* [Issue 244](https://github.com/pytroll/satpy/issues/244) - Fix NUCAPS reader for NUCAPS EDR v2 files ([PR 326](https://github.com/pytroll/satpy/pull/326))
* [Issue 236](https://github.com/pytroll/satpy/issues/236) - scene.resample(cache_dir=) fails with TypeError: Unicode-objects must be encoded before hashing
* [Issue 233](https://github.com/pytroll/satpy/issues/233) - IOError: Unable to read attribute (no appropriate function for conversion path)
* [Issue 211](https://github.com/pytroll/satpy/issues/211) - Fix OLCI and other readers' file patterns to work on Windows
* [Issue 207](https://github.com/pytroll/satpy/issues/207) - Method not fully documented in terms of possible key word arguments
* [Issue 199](https://github.com/pytroll/satpy/issues/199) - Reading Modis file produce a double image
* [Issue 168](https://github.com/pytroll/satpy/issues/168) - Cannot read MODIS data
* [Issue 167](https://github.com/pytroll/satpy/issues/167) - KeyError 'v' using Scene(base_dir=, reader=) ([PR 325](https://github.com/pytroll/satpy/pull/325))
* [Issue 165](https://github.com/pytroll/satpy/issues/165) - HRIT GOES reader is broken ([PR 303](https://github.com/pytroll/satpy/pull/303))
* [Issue 160](https://github.com/pytroll/satpy/issues/160) - Inconsistent naming of optional datasets in composite configs and compositors
* [Issue 157](https://github.com/pytroll/satpy/issues/157) - Add animation example ([PR 322](https://github.com/pytroll/satpy/pull/322))
* [Issue 156](https://github.com/pytroll/satpy/issues/156) - Add cartopy example
* [Issue 146](https://github.com/pytroll/satpy/issues/146) - Add default null log handler
* [Issue 123](https://github.com/pytroll/satpy/issues/123) - NetCDF writer doesn't work ([PR 307](https://github.com/pytroll/satpy/pull/307))
* [Issue 114](https://github.com/pytroll/satpy/issues/114) - Print a list of available sensors/readers
* [Issue 82](https://github.com/pytroll/satpy/issues/82) - Separate file discovery from Scene init
* [Issue 61](https://github.com/pytroll/satpy/issues/61) - Creating composites post-load
* [Issue 10](https://github.com/pytroll/satpy/issues/10) - Optimize CREFL for memory

In this release 24 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 331](https://github.com/pytroll/satpy/pull/331) - Adapt slstr reader to xarray&dask
* [PR 329](https://github.com/pytroll/satpy/pull/329) - issue#328: fixed bugs loading JMA HRIT files ([328](https://github.com/pytroll/satpy/issues/328))
* [PR 326](https://github.com/pytroll/satpy/pull/326) - Fix nucaps reader for NUCAPS EDR v2 files ([244](https://github.com/pytroll/satpy/issues/244), [244](https://github.com/pytroll/satpy/issues/244))
* [PR 325](https://github.com/pytroll/satpy/pull/325) - Fix exception when Scene is given reader and base_dir ([167](https://github.com/pytroll/satpy/issues/167))
* [PR 319](https://github.com/pytroll/satpy/pull/319) - Fix msi reader delayed
* [PR 318](https://github.com/pytroll/satpy/pull/318) - Fix nir reflectance to use XArray
* [PR 312](https://github.com/pytroll/satpy/pull/312) - Allow custom regions in ahi-hsd file patterns
* [PR 311](https://github.com/pytroll/satpy/pull/311) - Allow valid_range to be a tuple for cloud product colorization
* [PR 303](https://github.com/pytroll/satpy/pull/303) - Fix hrit goes to support python 3 ([165](https://github.com/pytroll/satpy/issues/165))
* [PR 288](https://github.com/pytroll/satpy/pull/288) - Fix hrit-goes reader
* [PR 192](https://github.com/pytroll/satpy/pull/192) - Clip day and night composites after enhancement

#### Features added

* [PR 315](https://github.com/pytroll/satpy/pull/315) - Add slicing to Scene
* [PR 314](https://github.com/pytroll/satpy/pull/314) - Feature mitiff writer
* [PR 307](https://github.com/pytroll/satpy/pull/307) - Fix projections in cf writer ([123](https://github.com/pytroll/satpy/issues/123))
* [PR 305](https://github.com/pytroll/satpy/pull/305) - Add support for geolocation and angles to msi reader
* [PR 302](https://github.com/pytroll/satpy/pull/302) - Workaround the LinearNDInterpolator thread-safety issue for Sentinel 1 SAR geolocation
* [PR 301](https://github.com/pytroll/satpy/pull/301) - Factorize header definitions between hrit_msg and native_msg. Fix a bug in header definition.
* [PR 298](https://github.com/pytroll/satpy/pull/298) - Implement sentinel 2 MSI reader
* [PR 294](https://github.com/pytroll/satpy/pull/294) - Add the ocean color product to olci
* [PR 153](https://github.com/pytroll/satpy/pull/153) - [WIP] Improve compatibility of cf_writer with CF-conventions

In this release 20 pull requests were closed.


## Version 0.9.0a2 (2018/05/14)

### Issues Closed

* [Issue 286](https://github.com/pytroll/satpy/issues/286) - Proposal: search automatically for local config-files/readers
* [Issue 278](https://github.com/pytroll/satpy/issues/278) - msg native reader fails on full disk image
* [Issue 277](https://github.com/pytroll/satpy/issues/277) - msg_native reader fails when order number has a hyphen in it ([PR 282](https://github.com/pytroll/satpy/pull/282))
* [Issue 270](https://github.com/pytroll/satpy/issues/270) - How to find the value at certain latitude and longtitude
* [Issue 269](https://github.com/pytroll/satpy/issues/269) - How to intepret the parameter values in  AreaDefinition
* [Issue 268](https://github.com/pytroll/satpy/issues/268) - How to find the appropriate values of parameters in Scene.resample() function using Himawari Data
* [Issue 241](https://github.com/pytroll/satpy/issues/241) - reader native_msg using `np.str`
* [Issue 218](https://github.com/pytroll/satpy/issues/218) - Resampling to EPSG:4326 produces unexpected results
* [Issue 189](https://github.com/pytroll/satpy/issues/189) - Error when reading MSG native format
* [Issue 62](https://github.com/pytroll/satpy/issues/62) - msg_native example
* [Issue 33](https://github.com/pytroll/satpy/issues/33) - Load metadata without loading data

In this release 11 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 290](https://github.com/pytroll/satpy/pull/290) - Fix unicode-named data loading
* [PR 285](https://github.com/pytroll/satpy/pull/285) - Fix native_msg calibration bug
* [PR 282](https://github.com/pytroll/satpy/pull/282) - Fix native_msg reader for ROI input and multi-part order file patterns ([277](https://github.com/pytroll/satpy/issues/277))
* [PR 280](https://github.com/pytroll/satpy/pull/280) - Fix CLAVR-x reader to work with xarray
* [PR 274](https://github.com/pytroll/satpy/pull/274) - Convert ahi hsd reader to dask and xarray
* [PR 265](https://github.com/pytroll/satpy/pull/265) - Bugfix msg native reader
* [PR 262](https://github.com/pytroll/satpy/pull/262) - Fix dependency tree to find the best dependency when multiple matches occur
* [PR 260](https://github.com/pytroll/satpy/pull/260) - Fix ABI L1B reader masking data improperly

#### Features added

* [PR 293](https://github.com/pytroll/satpy/pull/293) - Switch to netcdf4 as engine for nc nwcsaf reading
* [PR 292](https://github.com/pytroll/satpy/pull/292) - Use pyresample's boundary classes
* [PR 291](https://github.com/pytroll/satpy/pull/291) - Allow datasets without areas to be concatenated
* [PR 289](https://github.com/pytroll/satpy/pull/289) - Fix so UMARF files (with extention .nat) are found as well
* [PR 287](https://github.com/pytroll/satpy/pull/287) - Add production configuration for NWCSAF RDT, ASII products by Marco Sassi
* [PR 283](https://github.com/pytroll/satpy/pull/283) - Add GRIB Reader ([279](https://github.com/pytroll/satpy/issues/279))
* [PR 281](https://github.com/pytroll/satpy/pull/281) - Port the maia reader to dask/xarray
* [PR 276](https://github.com/pytroll/satpy/pull/276) - Support reducing data for geos areas ([272](https://github.com/pytroll/satpy/issues/272))
* [PR 273](https://github.com/pytroll/satpy/pull/273) - Msg readers cleanup ([267](https://github.com/pytroll/satpy/issues/267))
* [PR 271](https://github.com/pytroll/satpy/pull/271) - Add appveyor and use ci-helpers for CI environments
* [PR 264](https://github.com/pytroll/satpy/pull/264) - Add caching at the scene level, and handle saving/loading from disk
* [PR 262](https://github.com/pytroll/satpy/pull/262) - Fix dependency tree to find the best dependency when multiple matches occur

In this release 20 pull requests were closed.


## Version 0.9.0a1 (2018/04/22)

### Issues Closed

* [Issue 227](https://github.com/pytroll/satpy/issues/227) - Issue Reading MSG4
* [Issue 225](https://github.com/pytroll/satpy/issues/225) - Save Datasets using SCMI ([PR 228](https://github.com/pytroll/satpy/pull/228))
* [Issue 215](https://github.com/pytroll/satpy/issues/215) - Change `Scene.compute` to something else ([PR 220](https://github.com/pytroll/satpy/pull/220))
* [Issue 208](https://github.com/pytroll/satpy/issues/208) - Strange behaviour when trying to load data to a scene object after having worked with it ([PR 214](https://github.com/pytroll/satpy/pull/214))
* [Issue 200](https://github.com/pytroll/satpy/issues/200) - Different mask handling when saving to PNG or GeoTIFF ([PR 201](https://github.com/pytroll/satpy/pull/201))
* [Issue 176](https://github.com/pytroll/satpy/issues/176) - Loading viirs natural_color composite fails ([PR 177](https://github.com/pytroll/satpy/pull/177))

In this release 6 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 259](https://github.com/pytroll/satpy/pull/259) - Fix writer and refactor so bad writer name raises logical exception
* [PR 257](https://github.com/pytroll/satpy/pull/257) - Fix geotiff and png writers to save to a temporary directory
* [PR 256](https://github.com/pytroll/satpy/pull/256) - Add 'python_requires' to setup.py to specify python support
* [PR 253](https://github.com/pytroll/satpy/pull/253) - Fix ABI L1B reader to use 64-bit scaling factors for X/Y variables
* [PR 250](https://github.com/pytroll/satpy/pull/250) - Fix floating point geotiff saving in dask geotiff writer
* [PR 249](https://github.com/pytroll/satpy/pull/249) - Fix float geotiff saving on 0.8
* [PR 248](https://github.com/pytroll/satpy/pull/248) - Fix unloading composite deps when one of them has incompatible areas
* [PR 243](https://github.com/pytroll/satpy/pull/243) - Remove ABI composite reducerX modifiers

#### Features added

* [PR 252](https://github.com/pytroll/satpy/pull/252) - Use rasterio to save geotiffs when available
* [PR 239](https://github.com/pytroll/satpy/pull/239) - Add CSPP Geo (geocat) AHI reading support

In this release 10 pull requests were closed.


## Version 0.9.0a0 (2018-03-20)

#### Bugs fixed

* [Issue 179](https://github.com/pytroll/satpy/issues/179) - Cannot read AVHRR in AAPP format
* [PR 234](https://github.com/pytroll/satpy/pull/234) - Bugfix sar reader
* [PR 231](https://github.com/pytroll/satpy/pull/231) - Bugfix palette based compositor concatenation
* [PR 230](https://github.com/pytroll/satpy/pull/230) - Fix dask angle calculations of rayleigh corrector
* [PR 229](https://github.com/pytroll/satpy/pull/229) - Fix bug in dep tree when modifier deps are modified wavelengths
* [PR 228](https://github.com/pytroll/satpy/pull/228) - Fix 'platform' being used instead of 'platform_name'
* [PR 224](https://github.com/pytroll/satpy/pull/224) - Add helper method for checking areas in compositors
* [PR 222](https://github.com/pytroll/satpy/pull/222) - Fix resampler caching by source area
* [PR 221](https://github.com/pytroll/satpy/pull/221) - Fix Scene loading and resampling when generate=False
* [PR 220](https://github.com/pytroll/satpy/pull/220) - Rename Scene's `compute` to `generate_composites`
* [PR 219](https://github.com/pytroll/satpy/pull/219) - Fixed native_msg calibration problem and added env var to change the …
* [PR 214](https://github.com/pytroll/satpy/pull/214) - Fix Scene not being copied properly during resampling
* [PR 210](https://github.com/pytroll/satpy/pull/210) - Bugfix check if lons and lats should be masked before resampling
* [PR 206](https://github.com/pytroll/satpy/pull/206) - Fix optional dependencies not being passed to modifiers with opts only
* [PR 187](https://github.com/pytroll/satpy/pull/187) - Fix reader configs having mismatched names between filename and config
* [PR 185](https://github.com/pytroll/satpy/pull/185) - Bugfix nwcsaf_pps reader for file discoverability
* [PR 177](https://github.com/pytroll/satpy/pull/177) - Bugfix viirs loading - picked from (xarray)develop branch
* [PR 163](https://github.com/pytroll/satpy/pull/163) - Bugfix float geotiff

#### Features added

* [PR 232](https://github.com/pytroll/satpy/pull/232) - Add ABI L1B system tests
* [PR 226](https://github.com/pytroll/satpy/pull/226) - EARS NWCSAF products reading
* [PR 217](https://github.com/pytroll/satpy/pull/217) - Add xarray/dask support to DayNightCompositor
* [PR 216](https://github.com/pytroll/satpy/pull/216) - Fix dataset writing so computations are shared between tasks
* [PR 213](https://github.com/pytroll/satpy/pull/213) - [WIP] Reuse same resampler for similar datasets
* [PR 212](https://github.com/pytroll/satpy/pull/212) - Improve modis reader to support dask
* [PR 209](https://github.com/pytroll/satpy/pull/209) - Fix enhancements to work with xarray
* [PR 205](https://github.com/pytroll/satpy/pull/205) - Fix ABI 'natural' and 'true_color' composites to work with xarray
* [PR 204](https://github.com/pytroll/satpy/pull/204) - Add 'native' resampler
* [PR 203](https://github.com/pytroll/satpy/pull/203) - [WIP] Feature trollimage xarray
* [PR 195](https://github.com/pytroll/satpy/pull/195) - Add ABI-specific configs for Airmass composite
* [PR 186](https://github.com/pytroll/satpy/pull/186) - Add missing nodata tiff tag
* [PR 180](https://github.com/pytroll/satpy/pull/180) - Replace BW and RGBCompositor with a more generic one

#### Documentation changes

* [PR 155](https://github.com/pytroll/satpy/pull/155) - Add contributing and developers guide documentation

In this release 1 issue and 31 pull requests were closed.
