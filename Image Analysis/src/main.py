from preprocessing import load_data, normalize_image, denoise_image
from segmentation import segment_regions
from tracking import track_objects


def run_pipeline(path):
    data = load_data(path)
    data = normalize_image(data)
    data = denoise_image(data)

    masks = segment_regions(data)
    tracks = track_objects(masks)

    return tracks