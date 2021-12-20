import gc
from abc import ABC, abstractmethod

import rasterio
from s2cloudless import S2PixelCloudDetector

from Pipeline.logger import log
from Pipeline.Worker import S2Worker
from Pipeline.GranuleCalculator import GranuleCalculator
from Pipeline.Granule import S2Granule
import numpy as np
from Pipeline.Granule import S2Granule
from Pipeline.utils import *
from numba.typed import List as LIST
from Pipeline.Mask import S2JIT
from Pipeline.Detectors import S2Detectors
from Pipeline.Plotting import *
from Download.Sentinel2 import Downloader


class Task(ABC):

    @staticmethod
    @abstractmethod
    def perform_computation(worker: S2Worker, *args) -> S2Granule:
        raise NotImplemented


class NdviPerPixel(Task):

    @staticmethod
    def perform_computation(worker: S2Worker, constraint: int = 5) -> S2Granule:
        # log.info(f"Running optimised ndvi masking. Dataset {worker.main_dataset_path}")
        res_x, res_y = worker.get_res()
        # we don't need to stack all ndvi arrays, we need just the constraint and result
        ndvi_arrays = np.zeros(
            shape=(len(worker.granules) if len(worker.granules) < constraint else constraint, res_x, res_y),
            dtype=np.float
        )
        # this ndvi array serves as a holder of the current max ndvi value for this pixel
        ndvi_result = np.ones(shape=(res_x, res_y), dtype=np.float) * (-10)
        result = np.ones(shape=(len(worker.output_bands), res_x, res_y), dtype=np.uint16)
        doy = np.zeros(shape=(res_x, res_y), dtype=np.uint16)
        log.info(f"{(len(worker.granules) - 1) // constraint + 1} iteration(s) expected!")
        for iteration in range((len(worker.granules) - 1) // constraint + 1):
            # compute NDVI
            current_doy = LIST()
            current_data = LIST()
            log.info(f"Calculating NDVI arrays for iteration {iteration}")
            # Acquire first batch of granules, for instance constraint=4, granules=[0,1,2,3]
            workers = worker.granules[iteration * constraint: (iteration + 1) * constraint]
            for i, w in enumerate(workers, 0):
                current_doy.append(w.doy)
                # Prepare data
                w.load_bands()
                GranuleCalculator.s2_ndvi(w)
                # TODO: take mask as function (like per-tile)
                mask = (w["B02"] > 100) & (w["B04"] > 100) & (w["B8A"] > 500) & (w["B8A"] < 8000) & (w["AOT"] < 100)
                ndvi_arrays[i] = np.ma.array(w.temp["NDVI"], mask=~mask, fill_value=-1).filled()
                del mask
                current_data.append(w.stack_bands(worker.output_bands))
                w.free_resources()  # We have copied the resources to the new numpy array inside current_data
            S2JIT.s2_ndvi_pixel_analysis(ndvi_arrays, ndvi_result, current_data, current_doy, result, doy, res_x, res_y)
            log.debug(f"Done!")
            current_data = LIST()  # last iteration, del the pointer to the arrays

        # Init
        for i, band in enumerate(worker.output_bands, 0):
            worker.result[band] = result[i]
        worker.result["DOY"] = doy
        gc.collect()
        log.info("Saving result to the files...")
        worker._save_result()  # Save result into files
        r, g, b = extract_rgb_paths(worker.save_result_path)
        create_rgb_uint8(r, g, b, worker.save_result_path, worker.mercator)
        log.info("Done!")
        worker.release_bands()
        # If there's an intention to work further with the files
        # Return result Granule
        return S2Granule(worker.save_result_path, worker.spatial_resolution, worker.output_bands + ["rgb"])


class S2CloudlessPerPixel(Task):
    bands_l1c = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
    cloud_detector = S2PixelCloudDetector(
        threshold=0.4,
        average_over=4,
        dilation_size=2,
        all_bands=False
    )

    @staticmethod
    def perform_computation(worker: S2Worker, constraint: int = 10) -> S2Granule:
        """
        This method uses the s2cloudless algorithm provided by sentinel hub to mask the images.
        Uses the detector that is also used for the per-tile.
        :param worker: s2worker with data
        :param constraint: how many mask we allow to be opened at the same time
        :return: masked granule
        """
        res_x, res_y = worker.get_res()
        result = np.ones(shape=(len(worker.output_bands), res_x, res_y), dtype=np.uint16)
        doy = np.zeros(shape=(res_x, res_y), dtype=np.uint16)
        #  We will provide probability mask as the result as well
        final_mask = np.ones(shape=(res_x, res_y), dtype=np.int) * 255
        #  First thing, we will sort the granules based on their doy, so we get the latest result
        worker.granules.sort(key=lambda x: x.doy)
        #  Download the corresponding L1C datasets and compute the mask
        l1c_granules = [download_l1c(granule) for granule in worker.granules]
        # ^They are in the same order as worker granules

        # Create cloud masks
        for i, gl1c in enumerate(l1c_granules, 0):
            # Get cld prob
            path = create_cloud_product(gl1c)
            worker.granules[i].add_another_band(path, "CLD")

        #  Do the masking
        log.info(f"{(len(worker.granules) - 1) // constraint + 1} iteration(s) expected!")
        for iteration in range((len(worker.granules) - 1) // constraint + 1):
            current_doy = LIST()
            # mind these are probability masks !! -> might also be bin. mask depends on params
            # in per pixel it is better to work with probability masks
            current_masks = LIST()
            current_data = LIST()
            # Acquire first batch of granules, for instance constraint=4, granules=[0,1,2,3], second -> [4,5,6,7]
            granules = worker.granules[iteration * constraint: (iteration + 1) * constraint]
            for i, g in enumerate(granules, 0):
                current_doy.append(g.doy)
                current_masks.append(g["CLD"])
                current_data.append(g.stack_bands(worker.output_bands))
            S2JIT.s2_cloud_probability_analysis(current_data, current_masks, current_doy, result, doy, final_mask)
            del current_data
            del current_masks
        log.info("Masking done")
        gc.collect()
        # Init
        for i, band in enumerate(worker.output_bands, 0):
            worker.result[band] = result[i]
        worker.result["DOY"] = doy
        log.info("Saving result to the files...")
        worker._save_result()  # Save result into files
        r, g, b = extract_rgb_paths(worker.save_result_path)
        create_rgb_uint8(r, g, b, worker.save_result_path, worker.mercator)
        worker.release_bands()
        # If there's an intention to work further with the files
        return S2Granule(worker.save_result_path, worker.spatial_resolution, worker.output_bands + ["rgb"])


class MedianPerPixel(Task):

    @staticmethod
    def perform_computation(worker: S2Worker, args=None) -> S2Granule:
        """
        This method takes the median of all the pixels.
        """
        log.info(f"Running per-pixel median masking. Dataset {worker.main_dataset_path}")
        log.info(f"Picked bands: {worker.output_bands}, expected iterations: {len(worker.output_bands)}")
        res_x, res_y = worker.get_res()
        for i, band_key in enumerate(worker.output_bands, 0):
            #  Reference object for yielding size of window block, since the blocks might not be same in each iteration
            #  this is the best of possible ways to get the block
            reference_object = worker.granules[0][band_key].path
            worker.result[band_key] = np.ones(shape=(res_x, res_y), dtype=np.uint16)
            with rasterio.open(reference_object) as reference:
                for ji, window in reference.block_windows(1):
                    current_blocks = LIST()  # array of blocks where for each pixel median is picked
                    for j, granule in enumerate(worker.granules, 0):
                        current_blocks.append(granule[band_key].rasterio_ref().read(1, window=window))
                    data = np.stack(current_blocks)
                    median_values = np.median(data, axis=0)
                    res = S2JIT.s2_median_analysis(data, median_values)
                    del data
                    # current_blocks is filled now get the median
                    worker.result[band_key][window.row_off:window.row_off + window.height,
                    window.col_off:window.col_off + window.width] = res
            log.info(f"{band_key} done.")
        log.info("Saving result to the files...")
        worker._save_result()  # Save result into files
        r, g, b = extract_rgb_paths(worker.save_result_path)
        create_rgb_uint8(r, g, b, worker.save_result_path, worker.mercator)
        log.info("Done!")
        worker.release_bands()
        # If there's an intention to work further with the files
        # Return result Granule
        return S2Granule(worker.save_result_path, worker.spatial_resolution, worker.output_bands + ["rgb"])


class PerTile(Task):

    @staticmethod
    def perform_computation(worker: S2Worker, detector: S2Detectors = S2Detectors.scl) -> S2Granule:
        log.info(f"Running per-tile masking. Dataset {worker.main_dataset_path}")
        # Gather information
        res_x, res_y = s2_get_resolution(worker.spatial_resolution)
        slice_index = worker.slice_index

        # Check if might proceed to the next step which is per-tile procedure
        for _worker in worker.granules:
            if _worker.slice_index != slice_index:
                raise Exception("Terminating job. Workers with different slice index are not allowed!")

        # slice_raster(slice_index, res)  # For easier assignments
        doy = slice_raster(slice_index, np.zeros(shape=(res_x, res_y), dtype=np.uint16))
        log.warning(doy.size)
        # Result array, where we are going to store the result intensities of pixel
        # shape -> bands, slices, x, y
        res = np.zeros(shape=(len(worker.output_bands), doy.shape[0], doy.shape[1], doy.shape[2]), dtype=np.uint16)
        log.info(
            f"Initialized result array shape: {len(worker.output_bands), doy.shape[0], doy.shape[1], doy.shape[2]}")
        # using numpy for slicing features, could've been simple python 2D list as well
        cloud_info = np.zeros(shape=(len(worker.granules), slice_index * slice_index))

        # Imagine sit.: slice_index=5, func(w) returns [10,15,20,50,35], each of the number
        # corresponds to the cloud percentage of that area that was calculated with the 'func'
        # cloud_info[i] = func(w)
        for i, w in enumerate(worker.granules, 0):
            cloud_info[i] = GranuleCalculator.s2_pertile_cloud_index_mask(w, detector)

        # After iterations we hold 2D array where the y-axis stands for index of worker and
        # x-axis for the cloud percentage in the xth area of yth worker, now we just have to pick the one
        # with least cloud %
        # { worker_index: [slice_indices] }
        workers_to_use = {}
        for i in range(slice_index ** 2):
            winner = cloud_info[:, i].argmin()
            if winner in workers_to_use:
                workers_to_use[winner].append(i)
            else:
                workers_to_use[winner] = [i]
        for value in workers_to_use.keys():
            worker.granules[value].load_bands()
            stack = worker.granules[value].stack_bands(worker.output_bands)
            worker.granules[value].free_resources()
            log.info(f"Loading bands for worker with index: {value}, "
                     f"worker occupies slices - {workers_to_use[value]}")
            for sl_index in workers_to_use[value]:
                doy[sl_index] = worker.granules[value].doy
                res[:, sl_index, :, :] = stack[:, sl_index, :, :]
            stack = None
        worker.result["DOY"] = glue_raster(doy, res_y, res_x)
        result = np.zeros(shape=(len(worker.output_bands), res_x, res_y), dtype=np.uint16)
        for i in range(len(worker.output_bands)):
            result[i] = glue_raster(res[i, :, :, :], res_y, res_x)  # (25, 1098, 1098) => (5049, 5049)
        # Save it to the result
        for i, band in enumerate(worker.output_bands, 0):
            worker.result[band] = result[i]
        worker._save_result()
        r, g, b = extract_rgb_paths(worker.save_result_path)
        create_rgb_uint8(r, g, b, worker.save_result_path, worker.mercator)
        worker.release_bands()
        return S2Granule(worker.save_result_path, worker.spatial_resolution, worker.output_bands + ["rgb"])
