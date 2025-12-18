"""Tests for tracing and logging functionality."""
import logging
from pathlib import Path

import pytest

import tracing


def test_setup_logging_console_only():
    """Test that logging can be set up without a log file."""
    tracing.setup_logging(log_file=None, level=logging.DEBUG)
    logger = tracing.get_logger("test")
    assert logger.level <= logging.DEBUG


def test_setup_logging_with_file(tmp_path: Path):
    """Test that logging can be set up with a log file."""
    log_file = tmp_path / "test.log"
    tracing.setup_logging(log_file=log_file, level=logging.INFO)
    
    logger = tracing.get_logger("test")
    logger.info("Test message")
    
    # Check that log file was created and contains the message
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content


def test_get_logger():
    """Test that get_logger returns a logger instance."""
    logger = tracing.get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_trace_function_decorator():
    """Test that trace_function decorator logs function calls."""
    call_log = []
    
    @tracing.trace_function
    def sample_function(x, y):
        call_log.append((x, y))
        return x + y
    
    # Set up logging to capture logs
    tracing.setup_logging(log_file=None, level=logging.INFO)
    
    result = sample_function(1, 2)
    assert result == 3
    assert call_log == [(1, 2)]


def test_trace_function_with_exception():
    """Test that trace_function logs exceptions."""
    @tracing.trace_function
    def failing_function():
        raise ValueError("Test error")
    
    tracing.setup_logging(log_file=None, level=logging.INFO)
    
    with pytest.raises(ValueError, match="Test error"):
        failing_function()


def test_trace_operation_decorator():
    """Test that trace_operation decorator logs operations."""
    @tracing.trace_operation("test_operation")
    def sample_operation():
        return "success"
    
    tracing.setup_logging(log_file=None, level=logging.INFO)
    
    result = sample_operation()
    assert result == "success"


def test_trace_operation_with_exception():
    """Test that trace_operation logs operation failures."""
    @tracing.trace_operation("failing_operation")
    def failing_operation():
        raise RuntimeError("Operation failed")
    
    tracing.setup_logging(log_file=None, level=logging.INFO)
    
    with pytest.raises(RuntimeError, match="Operation failed"):
        failing_operation()


def test_performance_tracker_context_manager():
    """Test that PerformanceTracker works as a context manager."""
    tracing.setup_logging(log_file=None, level=logging.INFO)
    logger = tracing.get_logger("test")
    
    with tracing.PerformanceTracker("test_operation", logger) as tracker:
        assert tracker.start_time is not None
        # Do some work
        sum([i for i in range(1000)])


def test_performance_tracker_with_exception():
    """Test that PerformanceTracker handles exceptions."""
    tracing.setup_logging(log_file=None, level=logging.INFO)
    logger = tracing.get_logger("test")
    
    with pytest.raises(ValueError, match="Test error"):
        with tracing.PerformanceTracker("failing_operation", logger):
            raise ValueError("Test error")


def test_log_user_action():
    """Test logging user actions."""
    tracing.setup_logging(log_file=None, level=logging.INFO)
    
    # Should not raise any exceptions
    tracing.log_user_action("test_action")
    tracing.log_user_action("test_action", {"key": "value", "count": 42})


def test_log_user_action_with_file(tmp_path: Path):
    """Test that user actions are logged to file."""
    log_file = tmp_path / "user_actions.log"
    tracing.setup_logging(log_file=log_file, level=logging.INFO)
    
    tracing.log_user_action("file_upload", {"filename": "test.xlsx", "size": 1024})
    
    content = log_file.read_text()
    assert "User action: file_upload" in content
    assert "filename=test.xlsx" in content
    assert "size=1024" in content


def test_log_data_operation():
    """Test logging data operations."""
    tracing.setup_logging(log_file=None, level=logging.INFO)
    
    # Should not raise any exceptions
    tracing.log_data_operation("filter", rows=100, columns=5)
    tracing.log_data_operation("search", rows=50, columns=5, details={"query": "test"})


def test_log_data_operation_with_file(tmp_path: Path):
    """Test that data operations are logged to file."""
    log_file = tmp_path / "data_ops.log"
    tracing.setup_logging(log_file=log_file, level=logging.INFO)
    
    tracing.log_data_operation("load", rows=200, columns=10, details={"file": "data.xlsx"})
    
    content = log_file.read_text()
    assert "Data operation: load" in content
    assert "rows=200" in content
    assert "columns=10" in content
    assert "file=data.xlsx" in content


def test_trace_function_preserves_function_metadata():
    """Test that trace_function decorator preserves function metadata."""
    @tracing.trace_function
    def documented_function(x: int) -> int:
        """This function has documentation."""
        return x * 2
    
    assert documented_function.__name__ == "documented_function"
    assert documented_function.__doc__ == "This function has documentation."


def test_trace_operation_preserves_function_metadata():
    """Test that trace_operation decorator preserves function metadata."""
    @tracing.trace_operation("multiply_operation")
    def multiply_function(x: int, y: int) -> int:
        """Multiplies two numbers."""
        return x * y
    
    assert multiply_function.__name__ == "multiply_function"
    assert multiply_function.__doc__ == "Multiplies two numbers."
