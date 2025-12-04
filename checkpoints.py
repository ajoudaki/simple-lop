import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch


def get_run_id(args: Any, prefix: str = "run") -> str:
    """
    Generate a unique run identifier based on key config parameters and timestamp.
    """
    # Create a config string from key parameters
    config_str = f"{args.optimizer}_lr{args.lr}_h{args.h}_L{args.L}_{args.activation}_tasks{args.tasks}_steps{args.steps}"
    
    # Add a short hash for uniqueness
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{prefix}_{config_str}_{timestamp}_{config_hash}"


def create_output_dir(base_dir: str = "out", run_id: Optional[str] = None) -> Path:
    """
    Create an output directory for a run. Does not overwrite existing directories.
    
    Args:
        base_dir: Base directory for all outputs
        run_id: Unique identifier for this run
        
    Returns:
        Path to the created output directory
    """
    base_path = Path(base_dir)
    
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    
    output_dir = base_path / run_id
    
    # Ensure we don't overwrite - add suffix if needed
    counter = 1
    original_dir = output_dir
    while output_dir.exists():
        output_dir = Path(f"{original_dir}_{counter}")
        counter += 1
    
    output_dir.mkdir(parents=True, exist_ok=False)
    
    return output_dir


def create_checkpoint_dir(base_dir: str = "model_checkpoints", run_id: Optional[str] = None) -> Path:
    """
    Create a checkpoint directory for a run. Does not overwrite existing directories.
    
    Args:
        base_dir: Base directory for all checkpoints
        run_id: Unique identifier for this run
        
    Returns:
        Path to the created checkpoint directory
    """
    base_path = Path(base_dir)
    
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    
    checkpoint_dir = base_path / run_id
    
    # Ensure we don't overwrite - add suffix if needed
    counter = 1
    original_dir = checkpoint_dir
    while checkpoint_dir.exists():
        checkpoint_dir = Path(f"{original_dir}_{counter}")
        counter += 1
    
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    
    return checkpoint_dir


def save_config(output_dir: Path, args: Any, filename: str = "config.json") -> None:
    """
    Save the run configuration to a JSON file.
    """
    config_path = output_dir / filename
    
    # Convert args namespace to dict
    config_dict = vars(args).copy()
    
    # Convert non-serializable types
    for key, value in config_dict.items():
        if isinstance(value, (list, tuple)):
            config_dict[key] = list(value)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    task_idx: int,
    step_idx: int,
    gradients: Optional[Dict[str, torch.Tensor]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a model checkpoint and optionally the gradients.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        model: The model to checkpoint
        task_idx: Current task index (0-based)
        step_idx: Current step index within the task (0-based)
        gradients: Optional dict of parameter name -> gradient tensor
        additional_info: Optional additional metadata to save
    """
    # Create filename with task and step info
    checkpoint_name = f"task{task_idx:04d}_step{step_idx:04d}"
    
    # Save model state
    model_path = checkpoint_dir / f"{checkpoint_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save gradients if provided
    if gradients is not None:
        grad_path = checkpoint_dir / f"{checkpoint_name}_gradients.pt"
        torch.save(gradients, grad_path)
    
    # Save additional info if provided
    if additional_info is not None:
        info_path = checkpoint_dir / f"{checkpoint_name}_info.json"
        # Convert tensors to floats for JSON serialization
        serializable_info = {}
        for k, v in additional_info.items():
            if isinstance(v, torch.Tensor):
                serializable_info[k] = v.item() if v.numel() == 1 else v.tolist()
            else:
                serializable_info[k] = v
        with open(info_path, 'w') as f:
            json.dump(serializable_info, f, indent=2)


def get_model_gradients(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract current gradients from all model parameters.
    
    Returns:
        Dict mapping parameter names to their gradient tensors (cloned)
    """
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone().detach().cpu()
    return gradients


class OutputManager:
    """
    Manages output directories and file saving for experiment runs.
    """
    
    def __init__(self, args: Any, base_dir: str = "out"):
        """
        Initialize the output manager.
        
        Args:
            args: Command line arguments
            base_dir: Base directory for outputs
        """
        self.run_id = get_run_id(args)
        self.output_dir = create_output_dir(base_dir, self.run_id)
        save_config(self.output_dir, args)
        print(f"Outputs will be saved to: {self.output_dir}")
    
    def get_path(self, filename: str) -> Path:
        """Get full path for a file in the output directory."""
        return self.output_dir / filename
    
    def get_run_id(self) -> str:
        """Return the run ID."""
        return self.run_id
    
    def get_output_dir(self) -> Path:
        """Return the output directory path."""
        return self.output_dir


class CheckpointManager:
    """
    Manages checkpointing during training runs.
    """
    
    def __init__(
        self,
        args: Any,
        checkpoint_freq: int = 0,
        base_dir: str = "model_checkpoints",
        enabled: bool = True,
        run_id: Optional[str] = None
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            args: Command line arguments
            checkpoint_freq: Save checkpoint every N steps (0 = disabled)
            base_dir: Base directory for checkpoints
            enabled: Whether checkpointing is enabled
            run_id: Optional run ID to use (for consistency with OutputManager)
        """
        self.enabled = enabled and checkpoint_freq > 0
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = None
        
        if self.enabled:
            # Use provided run_id or generate a new one
            if run_id is None:
                run_id = get_run_id(args)
            self.checkpoint_dir = create_checkpoint_dir(base_dir, run_id)
            save_config(self.checkpoint_dir, args)
            print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
    
    def should_checkpoint(self, step_idx: int) -> bool:
        """Check if we should save a checkpoint at this step."""
        if not self.enabled:
            return False
        return step_idx % self.checkpoint_freq == 0
    
    def save(
        self,
        model: torch.nn.Module,
        task_idx: int,
        step_idx: int,
        save_gradients: bool = True,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a checkpoint if enabled and at the right frequency.
        """
        if not self.enabled:
            return
        
        gradients = get_model_gradients(model) if save_gradients else None
        
        save_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            model=model,
            task_idx=task_idx,
            step_idx=step_idx,
            gradients=gradients,
            additional_info=additional_info
        )
    
    def get_checkpoint_dir(self) -> Optional[Path]:
        """Return the checkpoint directory path."""
        return self.checkpoint_dir
