import sunpy.map
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import warnings
from astropy.wcs.utils import wcs_to_celestial_frame
from skimage import io
import os


def AreaC(observation, mask_image_path):
    # Check the rotation matrix
    m = np.array(observation.rotation_matrix, dtype=int).flatten()
    if set(m) != set([1, 0, 0, 1]):
        warnings.warn(
            'The rotation matrix corresponds to a non-zero degrees rotation. Please mind the rotation. An unrotated observation could cause confusion.')

    # Load the binary mask image
    mask = io.imread(mask_image_path, as_gray=True)
    mask = mask > 0  # Convert to binary mask
    mask = np.array(mask, dtype=int)

    # Check the input mask
    if np.min(mask) < 0 or np.max(mask) > 1:
        raise ValueError('The input mask must only contain True-False or zero-one pairs.')

    ny, nx = np.shape(mask)

    # Create a grid in arcsec
    xrange = np.linspace(-observation.rsun_obs.value, observation.rsun_obs.value, nx) * u.arcsec
    yrange = np.linspace(-observation.rsun_obs.value, observation.rsun_obs.value, ny) * u.arcsec

    xv, yv = np.meshgrid(xrange, yrange, indexing='xy')

    # Coordinate transformation from arcsec grid to Stonyhurst B and L (LCM)
    c = SkyCoord(xv, yv, frame=wcs_to_celestial_frame(observation.wcs))
    c = c.heliographic_stonyhurst

    # Results storing, distinguished by B and L
    b_mask = c.lat.value
    l_mask = c.lon.value

    # The difference between i-th and i+1-th pixels in latitude and lcm.
    b_diff = np.absolute(b_mask - np.roll(b_mask, 1, axis=0))
    l_diff = np.absolute(l_mask - np.roll(l_mask, 1, axis=1))

    # Whole mask area
    A0 = np.nansum(b_diff * l_diff * mask)
    A0 = A0 * (u.deg ** 2)

    # Convert to MSH: (spot in square degrees / MSH_unit)
    MSH_unit = 2 * np.pi * np.power((180 / np.pi), 2) / 1000000 * (u.deg ** 2)
    A1 = A0 / MSH_unit

    # Sphere area in km2
    S = (4 * np.pi * np.power(695700 * u.km, 2))
    S1 = (4 * np.pi * np.power(((180 * u.deg) / np.pi), 2))

    # Convert feature size to km2
    A2 = S * (A0 / S1)

    return [A0, A1, A2]


# def batch_process(fits_dir, mask_dir, output_csv, output_txt):
def batch_process(fits_dir, mask_dir, output_txt):
    # Collect all FITS and PNG files
    fits_files = [f for f in os.listdir(fits_dir) if f.endswith('.fits')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    results = []
    txt_results = []

    for mask_file in mask_files:
        # Extract the date part from the mask file name
        date_part = mask_file.split('.')[4][:8]  # Assumes the date part is the 4th element
        date_part_x = mask_file.split('.')[4][:8]+mask_file[-6:-4]  # Assumes the date part is the 4th element
        matching_fits_files = [f for f in fits_files if date_part in f]

        if not matching_fits_files:
            print(f"No matching FITS file found for {mask_file}")
            continue

        fits_file = matching_fits_files[0]  # Assuming the first match is the correct one

        # Construct the full file paths
        fits_path = os.path.join(fits_dir, fits_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # Load the SunPy map
        observation = sunpy.map.Map(fits_path)

        # Calculate the area
        areas = AreaC(observation, mask_path)

        # Append the result
        results.append((date_part_x, areas[1].value))

        # Extract the date and MSH

        txt_results.append((date_part_x, areas[1].value))

        # Output the result
        print(
            f"Processed {fits_file}: Area in square degrees = {areas[0]}, MSH = {areas[1]}, Area in square kilometers = {areas[2]}")

    # Save results to a CSV file
    import csv
    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['FITS File',  'Area (MSH)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(
                {'FITS File': result[0],  'Area (MSH)': result[1]})

    # Save date and MSH results to a TXT file
    with open(output_txt, 'a') as txtfile:
        for date, msh in txt_results:
            txtfile.write(f"{date} {msh}\n")


# Example usage:
fits_dir = 'D:\SunSpotData\original_data\hmi_ic_4k'  # replace with your actual directory path
mask_dir = 'box_all_mengban'  # replace with your actual directory path
output_csv = 'box_all_mengban/boxmengban_all_area.csv'  # replace with your desired output CSV file path
output_txt = 'box_all_mengban/boxmengban_all_area.txt'  # replace with your desired output TXT file path


# batch_process(fits_dir, mask_dir, output_csv, output_txt)
batch_process(fits_dir, mask_dir, output_txt)
