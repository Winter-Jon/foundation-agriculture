import asyncio
import sys
import logging
import json
import aiofiles
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict

from ..config import PipelineConfig
from ..dataset import BaseDataset
from ..agents.registry import AgentRegistry
from ..utils.importer import import_modules_from_folder

logger = logging.getLogger(__name__)

async def run_batch_task(
    registry: AgentRegistry,
    dataset: BaseDataset,
    task_name: str,
    config: PipelineConfig
) -> Dict[str, Any]:
    
    # 获取 Agent
    agent = registry.get_agent(task_name)
    if not agent:
        raise ValueError(f"Agent for task '{task_name}' not found in registry.")

    logger.info(f">>> 开始执行: 数据集:{dataset.name} 任务: {task_name} (并发数: {config.max_concurrent})")
    save_dir = config.save_dir / f"{dataset.name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization output directory for this task
    vis_output_dir = save_dir / task_name / "visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set visualization directory on all tools in the agent (if it's a ToolAgent)
    if hasattr(agent, 'tools'):
        for tool_name, tool in agent.tools._tools.items():
            if hasattr(tool, 'vis_output_dir'):
                tool.vis_output_dir = vis_output_dir
                logger.info(f"Set vis_output_dir for tool '{tool_name}': {vis_output_dir}")
    
    
    semaphore = asyncio.Semaphore(config.max_concurrent)
    results = {}
    
    async def worker(index: int):
        async with semaphore:
            item = await dataset.get_item(index)
            if not item.valid:
                return None
            
            try:
                # 获取 generation kwargs
                # 假设 TaskConfig 没有 generation_kwargs，我们从 ModelConfig 取
                # 或者 Agent 自己管理。这里从 config.model.generation_kwargs 取
                gen_kwargs = config.model.generation_kwargs or {}
                
                return await agent.run(item, config.model.model_name, **gen_kwargs)
            except Exception as e:
                logger.error(f"Worker Error {item.img_name}: {e}")
                return None


    # Create tasks for all items - they will respect the semaphore
    # Important: We must use create_task to schedule them on the loop immediately
    # otherwise they would run sequentially if we just awaited them one by one.
    worker_tasks = [asyncio.create_task(worker(i)) for i in range(len(dataset))]
    
    from tqdm.asyncio import tqdm
    task_results_list = []
    
    # Use as_completed to iterate over tasks as they finish
    # dynamic_ncols=True adapts to terminal width, mininterval avoids too frequent updates
    for future in tqdm(asyncio.as_completed(worker_tasks), desc=f"Processing {task_name}", total=len(worker_tasks), dynamic_ncols=True, mininterval=1.0):
        task_results_list.append(await future)

    
    for res in task_results_list:
        if res:
            results.update(res)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for res in results.values():
        if res and "usage" in res:
            u = res["usage"]
            total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += u.get("completion_tokens", 0)
            total_usage["total_tokens"] += u.get("total_tokens", 0)

    # 保存结果
    task_file = save_dir / f"{task_name}_results.json"
    async with aiofiles.open(task_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=2))
        
    def check_success(v):
        if not v:
            return False
        # Check top-level result or metadata result
        return v.get('result') or v.get('metadata', {}).get('result')

    success = sum(1 for v in results.values() if check_success(v))
    failed = len(results) - success
    
    stats = {
        "total_success": success,
        "total_failed": failed,
        "token_usage": total_usage
    }

    logger.info(f"任务 {task_name} 完成: 成功 {success}, 失败 {failed}, Tokens: {total_usage['total_tokens']}")

    return {
        "output_dir": str(save_dir),
        "results": results,
        "stats": stats,
        # "config": asdict(config) # PipelineConfig might not be serializable easily if it has complex types
    }
