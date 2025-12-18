"""
Tracing and logging utilities for the IoT Forecast Dashboard.

This module provides structured logging and performance monitoring
for key application operations.
"""
import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

# Configure logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
LOG_LEVEL = logging.INFO
LOG_FILE = Path("logs/app.log")


def setup_logging(log_file: Optional[Path] = None, level: int = LOG_LEVEL) -> None:
    """
    Set up application-wide logging configuration.
    
    Args:
        log_file: Optional path to log file. If None, logs to console only.
        level: Logging level (default: INFO)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.info("Logging initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def trace_function(func: Callable) -> Callable:
    """
    Decorator to trace function calls with timing information.
    
    Logs function entry, exit, execution time, and any exceptions.
    
    Args:
        func: Function to trace
    
    Returns:
        Wrapped function with tracing
    """
    logger = logging.getLogger(func.__module__)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        func_name = func.__name__
        logger.info(f"Entering {func_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"Exiting {func_name} - took {elapsed_time:.3f}s")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"Exception in {func_name} after {elapsed_time:.3f}s: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise
    
    return wrapper


def trace_operation(operation_name: str) -> Callable:
    """
    Decorator to trace specific operations with custom names.
    
    Args:
        operation_name: Descriptive name for the operation
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.info(f"Starting operation: {operation_name}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"Completed operation: {operation_name} - took {elapsed_time:.3f}s")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(
                    f"Failed operation: {operation_name} after {elapsed_time:.3f}s - "
                    f"{type(e).__name__}: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


class PerformanceTracker:
    """
    Context manager for tracking performance of code blocks.
    
    Example:
        with PerformanceTracker("data loading"):
            df = load_data()
    """
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        """
        Initialize performance tracker.
        
        Args:
            operation: Name of the operation being tracked
            logger: Optional logger instance (defaults to root logger)
        """
        self.operation = operation
        self.logger = logger or logging.getLogger()
        self.start_time = None
    
    def __enter__(self):
        self.logger.info(f"Starting: {self.operation}")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        if exc_type is not None:
            self.logger.error(
                f"Failed: {self.operation} after {elapsed_time:.3f}s - "
                f"{exc_type.__name__}: {exc_val}"
            )
        else:
            self.logger.info(f"Completed: {self.operation} in {elapsed_time:.3f}s")


def log_user_action(action: str, details: Optional[dict] = None) -> None:
    """
    Log a user action with optional details.
    
    Args:
        action: Description of the user action
        details: Optional dictionary of additional details
    """
    logger = logging.getLogger("user_actions")
    message = f"User action: {action}"
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        message += f" | {detail_str}"
    logger.info(message)


def log_data_operation(operation: str, rows: int, columns: int, details: Optional[dict] = None) -> None:
    """
    Log a data operation with row/column counts.
    
    Args:
        operation: Type of operation (e.g., "filter", "search")
        rows: Number of rows affected
        columns: Number of columns in dataset
        details: Optional additional details
    """
    logger = logging.getLogger("data_operations")
    message = f"Data operation: {operation} | rows={rows}, columns={columns}"
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        message += f" | {detail_str}"
    logger.info(message)
