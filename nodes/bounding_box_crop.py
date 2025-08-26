import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.hub import download_url_to_file
from safetensors.torch import load_file

import folder_paths
import comfy.model_management

try:
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.models import build_model
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util import box_ops
    from groundingdino.datasets.transforms import Compose, RandomResize, ToTensor, Normalize
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    GROUNDINGDINO_AVAILABLE = False
    print("Warning: GroundingDINO not available. DINO Crop node will not work.")

# Utility functions (previously from AILab_ImageMaskTools)
def pil2tensor(image):
    """Convert PIL Image to tensor"""
    import torch
    import numpy as np
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(tensor):
    """Convert tensor to PIL Image"""
    from PIL import Image
    import numpy as np
    # Remove batch dimension if present
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    # Convert to numpy and scale to 0-255
    if len(tensor.shape) == 3:
        np_image = (tensor.permute(2, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        if np_image.shape[0] == 1:
            return Image.fromarray(np_image[0], mode='L')
        elif np_image.shape[0] == 3:
            return Image.fromarray(np_image.transpose(1, 2, 0), mode='RGB')
        elif np_image.shape[0] == 4:
            return Image.fromarray(np_image.transpose(1, 2, 0), mode='RGBA')
    else:
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)

# GroundingDINO model definitions
DINO_MODELS = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/groundingdino_swint_ogc.safetensors",
        "config_filename": "GroundingDINO_SwinT_OGC.cfg.py",
        "model_filename": "groundingdino_swint_ogc.safetensors"
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.safetensors",
        "config_filename": "GroundingDINO_SwinB.cfg.py",
        "model_filename": "groundingdino_swinb_cogcoor.safetensors"
    }
}

def get_or_download_model_file(filename, url, dirname):
    local_path = folder_paths.get_full_path(dirname, filename)
    if local_path:
        return local_path
    folder = os.path.join(folder_paths.models_dir, dirname)
    os.makedirs(folder, exist_ok=True)
    local_path = os.path.join(folder, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from {url} ...")
        download_url_to_file(url, local_path)
    return local_path


class BoundingBoxCrop:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "prompt": "Enter text description of object to detect and crop",
            "dino_model": "GroundingDINO model for text-to-box detection",
            "threshold": "Detection threshold (higher = more strict)",
            "crop_mode": "How to handle multiple detections",
            "output_mode": "Crop: extract the object, Paint: paint out everything else",
            "padding": "Padding around the bounding box in pixels",
            "paint_color": "Color to paint outside the bounding box (hex format)",
            "device": "Auto: smart detection, CPU: force CPU, GPU: force GPU",
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Object to detect", "tooltip": tooltips["prompt"]}),
                "dino_model": (list(DINO_MODELS.keys()), {"default": "GroundingDINO_SwinT_OGC (694MB)", "tooltip": tooltips["dino_model"]}),
                "output_mode": (["Crop", "Paint"], {"default": "Crop", "tooltip": tooltips["output_mode"]}),
                "device": (["Auto", "CPU", "GPU"], {"default": "Auto", "tooltip": tooltips["device"]}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.01, "tooltip": tooltips["threshold"]}),
                "crop_mode": (["First", "Best", "All", "Merge"], {"default": "First", "tooltip": tooltips["crop_mode"]}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1, "tooltip": tooltips["padding"]}),
                "paint_color": ("COLOR", {"default": "#000000", "tooltip": tooltips["paint_color"]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("OUTPUT_IMAGE", "ORIGINAL_WITH_BOXES", "BBOX_INFO", "MASK")
    FUNCTION = "detect_and_crop"
    CATEGORY = "zsnodes/image"

    def __init__(self):
        self.dino_model_cache = {}

    def detect_and_crop(self, image, prompt, dino_model, output_mode, device, threshold=0.35, crop_mode="First", padding=10, paint_color="#000000"):
        if not GROUNDINGDINO_AVAILABLE:
            raise RuntimeError("GroundingDINO is not installed. Please install it to use this node.")
        
        device_obj = comfy.model_management.get_torch_device()

        # Process batch images
        batch_size = image.shape[0] if len(image.shape) == 4 else 1
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        result_outputs = []
        result_annotated = []
        result_masks = []
        bbox_infos = []
        
        # Parse paint color
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return (r, g, b)
        
        for b in range(batch_size):
            img_pil = tensor2pil(image[b])
            img_np = np.array(img_pil.convert("RGB"))
            
            # Load GroundingDINO config and weights
            dino_info = DINO_MODELS[dino_model]
            config_path = get_or_download_model_file(dino_info["config_filename"], dino_info["config_url"], "grounding-dino")
            weights_path = get_or_download_model_file(dino_info["model_filename"], dino_info["model_url"], "grounding-dino")

            # Load and cache GroundingDINO model
            dino_key = (config_path, weights_path, device_obj)
            if dino_key not in self.dino_model_cache:
                args = SLConfig.fromfile(config_path)
                model = build_model(args)
                checkpoint = load_file(weights_path)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    checkpoint = clean_state_dict(checkpoint['model'])
                model.load_state_dict(checkpoint, strict=False)
                model.eval()
                model.to(device_obj)
                self.dino_model_cache[dino_key] = model
            dino = self.dino_model_cache[dino_key]

            # Preprocess image for DINO
            transform = Compose([
                RandomResize([800], max_size=1333),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor, _ = transform(img_pil.convert("RGB"), None)
            image_tensor = image_tensor.unsqueeze(0).to(device_obj)

            # Prepare text prompt
            text_prompt = prompt if prompt.endswith(".") else prompt + "."

            # Forward pass
            with torch.no_grad():
                outputs = dino(image_tensor, captions=[text_prompt])
            logits = outputs["pred_logits"].sigmoid()[0]
            boxes = outputs["pred_boxes"][0]

            # Filter boxes by threshold
            filt_mask = logits.max(dim=1)[0] > threshold
            boxes_filt = boxes[filt_mask]
            scores_filt = logits.max(dim=1)[0][filt_mask]
            
            # Handle case with no detected boxes
            if boxes_filt.shape[0] == 0:
                result_outputs.append(image[b:b+1])
                result_annotated.append(image[b:b+1])
                # Create empty mask
                empty_mask = torch.zeros((1, img_pil.size[1], img_pil.size[0]), dtype=torch.float32)
                result_masks.append(empty_mask)
                bbox_infos.append("No objects detected")
                continue

            # Convert boxes to xyxy
            H, W = img_pil.size[1], img_pil.size[0]
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_filt)
            boxes_xyxy = boxes_xyxy * torch.tensor([W, H, W, H], dtype=torch.float32, device=boxes_xyxy.device)
            boxes_xyxy = boxes_xyxy.cpu().numpy()

            # Create annotated image with bounding boxes
            from PIL import ImageDraw, ImageFont
            annotated = img_pil.copy()
            draw = ImageDraw.Draw(annotated)
            
            # Try to use a better font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            for i, (box, score) in enumerate(zip(boxes_xyxy, scores_filt)):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                label = f"{prompt}: {score:.2f}"
                draw.text((x1, y1 - 20), label, fill="red", font=font)
            
            # Determine bounding box based on crop_mode
            if crop_mode == "First":
                x1, y1, x2, y2 = boxes_xyxy[0]
                bbox_info = f"x1:{x1:.0f}, y1:{y1:.0f}, x2:{x2:.0f}, y2:{y2:.0f}, score:{scores_filt[0]:.3f}"
            elif crop_mode == "Best":
                best_idx = scores_filt.argmax()
                x1, y1, x2, y2 = boxes_xyxy[best_idx]
                bbox_info = f"x1:{x1:.0f}, y1:{y1:.0f}, x2:{x2:.0f}, y2:{y2:.0f}, score:{scores_filt[best_idx]:.3f}"
            elif crop_mode == "Merge":
                x1 = min(boxes_xyxy[:, 0])
                y1 = min(boxes_xyxy[:, 1])
                x2 = max(boxes_xyxy[:, 2])
                y2 = max(boxes_xyxy[:, 3])
                bbox_info = f"Merged: x1:{x1:.0f}, y1:{y1:.0f}, x2:{x2:.0f}, y2:{y2:.0f}, detections:{len(boxes_xyxy)}"
            elif crop_mode == "All":
                # For "All" mode, we'll handle multiple boxes differently
                pass
            
            # Process based on output mode
            if output_mode == "Crop":
                if crop_mode == "All":
                    # Create individual crops for each detection
                    all_crops = []
                    all_masks = []
                    bbox_info_list = []
                    for i, (box, score) in enumerate(zip(boxes_xyxy, scores_filt)):
                        x1, y1, x2, y2 = box
                        x1 = max(0, int(x1) - padding)
                        y1 = max(0, int(y1) - padding)
                        x2 = min(W, int(x2) + padding)
                        y2 = min(H, int(y2) + padding)
                        cropped = img_pil.crop((x1, y1, x2, y2))
                        all_crops.append(pil2tensor(cropped))
                        # Create mask for this crop
                        mask = np.zeros((H, W), dtype=np.float32)
                        mask[y1:y2, x1:x2] = 1.0
                        all_masks.append(mask)
                        bbox_info_list.append(f"Box{i}: x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}, score:{score:.3f}")
                    
                    # Stack all crops (resize to max size)
                    if all_crops:
                        max_h = max(c.shape[1] for c in all_crops)
                        max_w = max(c.shape[2] for c in all_crops)
                        resized_crops = []
                        for crop in all_crops:
                            crop_pil = tensor2pil(crop[0])
                            crop_resized = crop_pil.resize((max_w, max_h), Image.LANCZOS)
                            resized_crops.append(pil2tensor(crop_resized))
                        result_outputs.append(torch.cat(resized_crops, dim=0))
                        # Combine all masks
                        combined_mask = np.maximum.reduce(all_masks) if len(all_masks) > 1 else all_masks[0]
                        mask_tensor = torch.from_numpy(combined_mask).unsqueeze(0)
                        result_masks.append(mask_tensor)
                        bbox_info = "; ".join(bbox_info_list)
                else:
                    # Single crop mode
                    x1 = max(0, int(x1) - padding)
                    y1 = max(0, int(y1) - padding)
                    x2 = min(W, int(x2) + padding)
                    y2 = min(H, int(y2) + padding)
                    cropped = img_pil.crop((x1, y1, x2, y2))
                    result_outputs.append(pil2tensor(cropped))
                    # Create mask for the cropped area
                    mask = np.zeros((H, W), dtype=np.float32)
                    mask[y1:y2, x1:x2] = 1.0
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0)
                    result_masks.append(mask_tensor)
                    
            else:  # Paint mode
                # Create mask for the detected area(s)
                mask = np.zeros((H, W), dtype=np.float32)
                
                if crop_mode == "All":
                    # Paint out everything except all detected objects
                    bbox_info_list = []
                    for i, (box, score) in enumerate(zip(boxes_xyxy, scores_filt)):
                        x1, y1, x2, y2 = box
                        x1 = max(0, int(x1) - padding)
                        y1 = max(0, int(y1) - padding)
                        x2 = min(W, int(x2) + padding)
                        y2 = min(H, int(y2) + padding)
                        mask[y1:y2, x1:x2] = 1.0
                        bbox_info_list.append(f"Box{i}: x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}, score:{score:.3f}")
                    bbox_info = "; ".join(bbox_info_list)
                else:
                    # Single bounding box already determined
                    x1 = max(0, int(x1) - padding)
                    y1 = max(0, int(y1) - padding)
                    x2 = min(W, int(x2) + padding)
                    y2 = min(H, int(y2) + padding)
                    mask[y1:y2, x1:x2] = 1.0
                
                # Apply paint color to areas outside the mask
                painted = img_pil.copy()
                painted_np = np.array(painted)
                rgb_color = hex_to_rgb(paint_color)
                
                # Paint out the areas where mask is 0
                painted_np[mask == 0] = rgb_color
                painted_pil = Image.fromarray(painted_np.astype(np.uint8))
                
                result_outputs.append(pil2tensor(painted_pil))
                mask_tensor = torch.from_numpy(mask).unsqueeze(0)
                result_masks.append(mask_tensor)
            
            result_annotated.append(pil2tensor(annotated))
            bbox_infos.append(bbox_info)

        # Combine all batch results
        if len(result_outputs) == 0:
            empty_mask = torch.zeros((batch_size, img_pil.size[1], img_pil.size[0]), dtype=torch.float32)
            return (image, image, "No objects detected", empty_mask)
            
        return (torch.cat(result_outputs, dim=0), 
                torch.cat(result_annotated, dim=0),
                "\n".join(bbox_infos),
                torch.cat(result_masks, dim=0))


NODE_CLASS_MAPPINGS = {
    "ZS_BoundingBoxCrop": BoundingBoxCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZS_BoundingBoxCrop": "Bounding Box Crop (ZS)",
}