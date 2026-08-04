"""
VLooM Pipeline Module

Provides customizable pipeline architecture for data processing workflows.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..config import PipelineConfig
    from ..agents.registry import AgentRegistry
    from ..core.llm_client import LLMClient
    from ..dataset import BaseDataset

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Abstract base class for VLooM pipelines.
    
    Subclass this to create custom pipelines with different processing logic.
    """
    
    def __init__(
        self,
        config: "PipelineConfig",
        registry: "AgentRegistry",
        llm_client: "LLMClient"
    ):
        self.config = config
        self.registry = registry
        self.llm_client = llm_client
        self.results: Dict[str, Any] = {}
    
    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """Execute the pipeline. Must be implemented by subclasses."""
        pass
    
    async def pre_run(self) -> None:
        """Hook called before pipeline execution. Override for custom setup."""
        pass
    
    async def post_run(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called after pipeline execution. Override for custom cleanup/metrics."""
        return results
    
    def get_datasets(self) -> List["BaseDataset"]:
        """Create dataset instances from config."""
        from ..dataset import create_dataset
        return [create_dataset(ds_cfg) for ds_cfg in self.config.datasets]


class DefaultPipeline(BasePipeline):
    """Default pipeline: iterates over datasets and tasks sequentially."""
    
    async def run(self) -> Dict[str, Any]:
        """Execute the default pipeline workflow."""
        from .batch_runner import run_batch_task
        
        await self.pre_run()
        
        datasets = self.get_datasets()
        tasks = self.config.run_tasks
        all_results = {}
        
        for dataset in datasets:
            logger.info(f"Dataset {dataset.name} has {len(dataset)} items.")
            dataset_results = {}
            
            for task_name in tasks:
                if not self.registry.get_agent(task_name):
                    logger.error(f"Skipping unknown task: {task_name}")
                    continue
                
                task_result = await run_batch_task(
                    self.registry, dataset, task_name, self.config
                )
                dataset_results[task_name] = task_result
            
            all_results[dataset.name] = dataset_results
        
        self.results = all_results
        return await self.post_run(all_results)


def get_pipeline_class(class_path: Optional[str] = None) -> type:
    """Get pipeline class from class path string or return default.
    
    Args:
        class_path: Dot-separated path like 'evaluation.pipeline.EvalPipeline'
        
    Returns:
        Pipeline class (not instance)
    """
    if not class_path:
        return DefaultPipeline
    
    try:
        import importlib
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        logger.error(f"Failed to load pipeline class '{class_path}': {e}")
        logger.info("Falling back to DefaultPipeline")
        return DefaultPipeline
