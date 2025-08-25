from .custom_save_image import CustomSaveImage
from .bounding_box_crop import NODE_CLASS_MAPPINGS as BBOX_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BBOX_DISPLAY_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "ZS_SaveImage": CustomSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZS_SaveImage": "Save Image (ZS Custom)",
}

# Merge bounding box crop mappings
NODE_CLASS_MAPPINGS.update(BBOX_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BBOX_DISPLAY_MAPPINGS)