import gc
from abc import ABC, abstractmethod
from Pipeline.logger import log
from Pipeline.Worker import S2Worker
from Pipeline.GranuleCalculator import GranuleCalculator
from Pipeline.Granule import S2Granule
import numpy as np
from Pipeline.Granule import S2Granule
from Pipeline.utils import *
from numba.typed import List as LIST
from Pipeline.Mask import S2JIT


class Task(ABC):

    @staticmethod
    @abstractmethod
    def perform_computation(*args) -> S2Granule:
        raise NotImplemented


class NdviPerPixel(Task):

    @staticmethod
    def perform_computation(worker: S2Worker, constraint: int = 5) -> S2Granule:
        log.info(f"Running optimised ndvi masking. Dataset {worker.main_dataset_path}")
        res_x, res_y = s2_get_resolution(worker.spatial_resolution)
        # we don't need to stack all ndvi arrays, we need just the constraint and result
        ndvi_arrays = np.zeros(shape=(constraint, res_x, res_y), dtype=np.float)
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
                ndvi_arrays[i] = np.ma.array(w.temp["NDVI"], mask=mask, fill_value=0).filled()
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
        # If there's an intention to work further with the files
        # Return result Granule
        return S2Granule(worker.save_result_path, worker.spatial_resolution, worker.output_bands)


class PerTile(Task):

    @staticmethod
    def perform_computation(worker: S2Worker, detector=GranuleCalculator.s2_cloud_mask_scl) -> S2Granule:
        log.info(f"Running per-tile masking. Dataset {worker.main_dataset_path}")
        # Gather information
        res_x, res_y = s2_get_resolution(worker.spatial_resolution)
        slice_index = worker.granules[-1].slice_index

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
                log.info(f"Filling worker: {value} with doy: {worker.granules[value].doy} on slice index {sl_index} "
                         f"for all available bands")
                doy[sl_index] = worker.granules[value].doy
                res[:, sl_index, :, :] = stack[:, sl_index, :, :]
            del stack
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
        return S2Granule(worker.save_result_path, worker.spatial_resolution, worker.output_bands)
