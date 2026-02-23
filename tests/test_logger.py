"""Tests for logger module."""

from src.inference.logger import Colors, Logger, LogLevel


class TestColors:
    """Tests for Colors class."""

    def test_colors_defined(self):
        """Test that all color codes are defined."""
        assert Colors.RED.startswith("\033[")
        assert Colors.GREEN.startswith("\033[")
        assert Colors.RESET == "\033[0m"
        assert Colors.BOLD == "\033[1m"

    def test_bright_colors_defined(self):
        """Test bright color variants."""
        assert Colors.BRIGHT_RED.startswith("\033[")
        assert Colors.BRIGHT_GREEN.startswith("\033[")
        assert Colors.BRIGHT_CYAN.startswith("\033[")


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_levels(self):
        """Test all log levels are defined."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestLogger:
    """Tests for Logger class."""

    def test_logger_init(self, output_stream):
        """Test logger initialization."""
        logger = Logger(name="test", use_colors=True, output_stream=output_stream)
        assert logger.name == "test"
        assert logger.use_colors is True

    def test_logger_no_colors(self, output_stream):
        """Test logger without colors."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.info("test message")
        output = output_stream.getvalue()
        assert "test message" in output
        assert "\033[" not in output  # No color codes

    def test_logger_info(self, output_stream):
        """Test info level logging."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.info("info message")
        output = output_stream.getvalue()
        assert "[INFO]" in output
        assert "info message" in output

    def test_logger_warning(self, output_stream):
        """Test warning level logging."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.warning("warning message")
        output = output_stream.getvalue()
        assert "[WARNING]" in output
        assert "warning message" in output

    def test_logger_debug(self, output_stream):
        """Test debug level logging."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.debug("debug message")
        output = output_stream.getvalue()
        assert "[DEBUG]" in output
        assert "debug message" in output

    def test_logger_with_component(self, output_stream):
        """Test logging with component."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.info("test message", component="workflow")
        output = output_stream.getvalue()
        assert "[WORKFLOW]" in output

    def test_task_started(self, output_stream):
        """Test task started logging."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.task_started("my_task")
        output = output_stream.getvalue()
        assert "Task started" in output
        assert "my_task" in output

    def test_task_completed(self, output_stream):
        """Test task completed logging."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.task_completed("my_task")
        output = output_stream.getvalue()
        assert "Task completed" in output
        assert "my_task" in output

    def test_task_killed(self, output_stream):
        """Test task killed logging."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.task_killed("my_task")
        output = output_stream.getvalue()
        assert "Task killed" in output
        assert "[WARNING]" in output

    def test_manager_starting(self, output_stream):
        """Test manager starting logging."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.manager_starting(5)
        output = output_stream.getvalue()
        assert "Starting with" in output
        assert "5" in output

    def test_manager_exiting(self, output_stream):
        """Test manager exiting logging."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.manager_exiting()
        output = output_stream.getvalue()
        assert "Exiting" in output

    def test_separator(self, output_stream):
        """Test separator logging."""
        logger = Logger(use_colors=False, output_stream=output_stream)
        logger.separator("Test Section")
        output = output_stream.getvalue()
        assert "Test Section" in output
        assert "=" in output

    def test_colorize_enabled(self):
        """Test colorization when enabled."""
        logger = Logger(use_colors=True)
        result = logger._colorize("test", Colors.RED)
        assert Colors.RED in result
        assert Colors.RESET in result

    def test_colorize_disabled(self):
        """Test colorization when disabled."""
        logger = Logger(use_colors=False)
        result = logger._colorize("test", Colors.RED)
        assert result == "test"
        assert Colors.RED not in result
