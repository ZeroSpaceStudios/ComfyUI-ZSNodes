import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args

class CustomSaveImage:
    """
    Custom save node that uses filename_prefix directly as the filename
    without adding automatic counters and batch numbers like the default SaveImage node
    """
    
    def __init__(self):
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The images to save."
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Used directly as filename (without automatic counter/batch number)"
                }),
                "output_folder": ("STRING", {
                    "default": "output",
                    "tooltip": "The folder to save the images to."
                }),
            },
            "optional": {
                "caption": ("STRING", {
                    "forceInput": True,
                    "tooltip": "String to save as .txt file alongside the image"
                }),
                "caption_file_extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "The extension for the caption file."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "zsnodes/image"
    DESCRIPTION = "Saves images using filename_prefix directly as filename without auto-incrementing counters"

    def save_images(self, images, output_folder, filename_prefix="ComfyUI",
                   prompt=None, extra_pnginfo=None, caption=None,
                   caption_file_extension=".txt"):
        
        filename_prefix += self.prefix_append
        
        # Handle absolute and relative paths
        if os.path.isabs(output_folder):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            full_output_folder = output_folder
        else:
            # Use ComfyUI's output directory as base
            output_dir = folder_paths.get_output_directory()
            full_output_folder = os.path.join(output_dir, output_folder)
            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder, exist_ok=True)
        
        results = []
        saved_filenames = []
        
        for batch_number, image in enumerate(images):
            # Convert tensor to PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Add metadata if not disabled
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            
            # CUSTOM NAMING LOGIC: Use filename_prefix directly
            # Only add batch number if there are multiple images
            if len(images) > 1:
                # If batch, add batch number with underscore
                file = f"{filename_prefix}_{batch_number:03d}.png"
            else:
                # Single image, use prefix as-is with .png extension
                file = f"{filename_prefix}.png"
            
            # Save the image
            image_path = os.path.join(full_output_folder, file)
            img.save(
                image_path,
                pnginfo=metadata,
                compress_level=self.compress_level
            )
            
            saved_filenames.append(file)
            
            results.append({
                "filename": file,
                "subfolder": output_folder,
                "type": self.type
            })
            
            # Save caption if provided
            if caption is not None:
                caption_filename = file.replace('.png', caption_file_extension)
                caption_path = os.path.join(full_output_folder, caption_filename)
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
        
        # Return the last saved filename (or all if you modify return types)
        return (saved_filenames[-1] if saved_filenames else "",)