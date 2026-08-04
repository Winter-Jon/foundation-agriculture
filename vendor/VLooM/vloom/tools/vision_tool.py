from typing import Any, List, Dict, Union, Tuple
import logging
import tempfile
import uuid
from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from .base import BaseTool
from ..utils import coco_to_qwen_bbox
import asyncio
import functools

logger = logging.getLogger(__name__)


def draw_bboxes_on_image(img_path: Path, bboxes: List[List[float]], output_dir: Path = None) -> Path:
    """
    Draw bounding boxes on an image and save to a temporary file.
    Args:
        img_path: Path to the source image.
        bboxes: List of [x, y, w, h] bounding boxes.
        output_dir: Optional directory for output. Uses tempdir if None.
    Returns:
        Path to the visualized image.
    """
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((x, y - 10), f"Box {i+1}", fill=color)

    out_name = f"vis_{uuid.uuid4().hex[:8]}.jpg"
    out_path = output_dir / out_name
    img.save(out_path)
    return out_path


from .registry import ToolRegistry

@ToolRegistry.register_tool("vision_tool")
class VisionTool(BaseTool):
    """
    Vision Tool that returns bounding boxes for an image from COCO annotations.
    Can be configured with GT (train.json) or Vertical Model predictions (predict_all.json).
    """
    name: str = "vision_tool"
    # ... (rest of class remains same, cutting for brevity in replacement)
    description: str = "Get bounding boxes for objects in an image. Returns bboxes and a visualization."

    def __init__(self, annotation_path: Path, image_dir: Path = None, vis_output_dir: Path = None):
        self.annotation_path = Path(annotation_path)
        if not self.annotation_path.exists():
            raise ValueError(f"COCO annotation file not found: {annotation_path}")
        
        self.image_dir = Path(image_dir) if image_dir else self.annotation_path.parent / "images_final"
        self.vis_output_dir = Path(vis_output_dir) if vis_output_dir else None
        
        logger.info(f"Loading COCO annotations from {self.annotation_path}...")
        self.coco = COCO(str(self.annotation_path))
        # Build filename -> image_id lookup
        self._filename_to_id = {img['file_name']: img['id'] for img in self.coco.imgs.values()}
        logger.info(f"COCO annotations loaded. {len(self._filename_to_id)} images indexed.")

    async def execute(self, image_id: int = None, image_name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Get bounding boxes and visualization for the given image.
        Args:
            image_id: ID of the image (int).
            image_name: Filename of the image (alternative to image_id).
        Returns:
            {"bboxes": List[List[float]], "visualization": str (path)}
        """
        # Resolve image_id from name if needed
        if image_id is None and image_name:
            image_id = self._filename_to_id.get(image_name)
            if image_id is None:
                return {"error": f"Image name '{image_name}' not found in annotations."}
        
        if image_id is None:
            return {"error": "Must provide either image_id or image_name."}

        try:
            img_id = int(image_id)
        except ValueError:
            return {"error": f"Invalid image_id {image_id}"}

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Get image path and info
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_dir / img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        if not img_path.exists():
            return {"error": f"Image file not found: {img_path}"}
            
        bboxes_coco = [ann['bbox'] for ann in anns]
        bboxes_qwen = [coco_to_qwen_bbox(b, img_w, img_h) for b in bboxes_coco]
        
        # Generate visualization (using COCO format for drawing)
        loop = asyncio.get_running_loop()
        vis_path = await loop.run_in_executor(None, draw_bboxes_on_image, img_path, bboxes_coco, self.vis_output_dir)
        
        return {
            "bboxes": bboxes_qwen,  # Return Qwen format
            "visualization": str(vis_path)
        }

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_id": {
                            "type": "integer",
                            "description": "The ID of the image to analyze."
                        },
                        "image_name": {
                            "type": "string",
                            "description": "The filename of the image to analyze (alternative to image_id)."
                        }
                    }
                }
            }
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "annotation_path": {"type": "str", "help": "Path to the COCO JSON annotation file.", "required": True},
            "image_dir": {"type": "str", "help": "Path to the image directory.", "required": False},
            "vis_output_dir": {"type": "str", "help": "Path to save visualization images.", "required": False}
        }


@ToolRegistry.register_tool("proposal_tool")
class ProposalTool(BaseTool):
    """
    Proposal Tool that returns candidate bounding boxes from a vertical model (predict_all.json).
    Essentially the same as VisionTool but semantically for "proposals" rather than GT.
    """
    name: str = "proposal_tool"
    description: str = "Get candidate bounding boxes from a vertical detection model. Returns bboxes and visualization."

    def __init__(self, annotation_path: Path, image_dir: Path = None, vis_output_dir: Path = None):
        self.annotation_path = Path(annotation_path)
        if not self.annotation_path.exists():
            raise ValueError(f"Proposal annotation file not found: {annotation_path}")
        
        self.image_dir = Path(image_dir) if image_dir else self.annotation_path.parent / "images_final"
        self.vis_output_dir = Path(vis_output_dir) if vis_output_dir else None
        
        logger.info(f"Loading proposal annotations from {self.annotation_path}...")
        self.coco = COCO(str(self.annotation_path))
        self._filename_to_id = {img['file_name']: img['id'] for img in self.coco.imgs.values()}
        logger.info(f"Proposal annotations loaded. {len(self._filename_to_id)} images indexed.")

    async def execute(self, image_id: int = None, image_name: str = None, **kwargs) -> Dict[str, Any]:
        if image_id is None and image_name:
            image_id = self._filename_to_id.get(image_name)
            if image_id is None:
                return {"error": f"Image name '{image_name}' not found."}
        
        if image_id is None:
            return {"error": "Must provide either image_id or image_name."}

        try:
            img_id = int(image_id)
        except ValueError:
            return {"error": f"Invalid image_id {image_id}"}

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes_coco = [ann['bbox'] for ann in anns]
        categories = [ann.get('category_id') for ann in anns]
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_dir / img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        bboxes_qwen = [coco_to_qwen_bbox(b, img_w, img_h) for b in bboxes_coco]
        
        vis_path = None
        if img_path.exists():
            loop = asyncio.get_running_loop()
            vis_path = await loop.run_in_executor(None, draw_bboxes_on_image, img_path, bboxes_coco, self.vis_output_dir)
        
        return {
            "bboxes": bboxes_qwen,  # Return Qwen format
            "categories": categories,
            "visualization": str(vis_path) if vis_path else None
        }

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_id": {"type": "integer", "description": "The ID of the image."},
                        "image_name": {"type": "string", "description": "The filename of the image."}
                    }
                }
            }
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "annotation_path": {"type": "str", "help": "Path to the proposal JSON file.", "required": True},
            "image_dir": {"type": "str", "help": "Path to the image directory.", "required": False},
            "vis_output_dir": {"type": "str", "help": "Path to save visualization images.", "required": False}
        }


@ToolRegistry.register_tool("visualizer_tool")
class VisualizerTool(BaseTool):
    """
    Visualizer Tool that draws user-provided bounding boxes on an image.
    Used for self-correction tasks where the model provides its own predictions.
    """
    name: str = "visualizer_tool"
    description: str = "Draw bounding boxes on an image. Provide the image name and bboxes to visualize."

    def __init__(self, image_dir: Path, vis_output_dir: Path = None):
        self.image_dir = Path(image_dir)
        self.vis_output_dir = Path(vis_output_dir) if vis_output_dir else None
        
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        logger.info(f"VisualizerTool initialized. Image dir: {self.image_dir}")

    async def execute(self, image_name: str, bboxes: List[List[float]], **kwargs) -> Dict[str, Any]:
        """
        Draw provided bboxes on the image.
        Args:
            image_name: Filename of the image.
            bboxes: List of [x1, y1, x2, y2] bounding boxes in Qwen format (0-1000).
        Returns:
            {"visualization": str (path)}
        """
        img_path = self.image_dir / image_name
        if not img_path.exists():
            return {"error": f"Image not found: {img_path}"}
            
        # Get dimensions for conversion
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            
        # Convert Qwen (0-1000) to COCO pixels for drawing
        bboxes_coco = []
        for bbox in bboxes:
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                x1_px = x1 * img_w / 1000
                y1_px = y1 * img_h / 1000
                x2_px = x2 * img_w / 1000
                y2_px = y2 * img_h / 1000
                w_px = x2_px - x1_px
                h_px = y2_px - y1_px
                bboxes_coco.append([x1_px, y1_px, w_px, h_px])
            else:
                bboxes_coco.append(bbox)
        
        loop = asyncio.get_running_loop()
        vis_path = await loop.run_in_executor(None, draw_bboxes_on_image, img_path, bboxes_coco, self.vis_output_dir)
        return {"visualization": str(vis_path)}

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_name": {"type": "string", "description": "The filename of the image."},
                        "bboxes": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "List of [x1, y1, x2, y2] bounding boxes in Qwen format (normalized 0-1000) to draw."
                        }
                    },
                    "required": ["image_name", "bboxes"]
                }
            }
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "image_dir": {"type": "str", "help": "Path to the image directory.", "required": True},
            "vis_output_dir": {"type": "str", "help": "Path to save visualization images.", "required": False}
        }

