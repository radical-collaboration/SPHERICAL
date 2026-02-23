"""Tests for utils module."""

from src.inference.utils import ensure_dir, get_gpus_for_node, load_config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_defaults(self, temp_dir):
        """Test loading config with defaults when file doesn't exist."""
        config = load_config(str(temp_dir / "nonexistent.yaml"))

        assert "model_path" in config
        assert "num_services" in config
        assert config["num_services"] == 1
        assert config["num_batches"] == 100
        assert config["max_batch_tokens"] == 16000

    def test_load_config_from_file(self, temp_dir):
        """Test loading config from file."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
num_services: 2
num_batches: 50
custom_key: custom_value
""")

        config = load_config(str(config_file))

        assert config["num_services"] == 2
        assert config["num_batches"] == 50
        assert config["custom_key"] == "custom_value"
        # Defaults should still be present
        assert "model_path" in config

    def test_load_config_merge_with_defaults(self, temp_dir):
        """Test that user config merges with defaults."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("num_services: 4")

        config = load_config(str(config_file))

        assert config["num_services"] == 4  # User value
        assert config["max_batch_tokens"] == 16000  # Default value


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_ensure_dir_creates_new(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_directory"
        assert not new_dir.exists()

        result = ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_dir_clears_existing(self, temp_dir):
        """Test clearing existing directory contents."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        # Create some files
        (existing_dir / "file1.txt").write_text("content1")
        (existing_dir / "file2.txt").write_text("content2")
        subdir = existing_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")

        assert len(list(existing_dir.iterdir())) == 3

        result = ensure_dir(existing_dir)

        assert existing_dir.exists()
        assert len(list(existing_dir.iterdir())) == 0
        assert result == existing_dir

    def test_ensure_dir_nested(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "a" / "b" / "c"

        ensure_dir(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()


class TestGetGpusForNode:
    """Tests for get_gpus_for_node function."""

    def test_get_devices_with_explicit_count(self):
        """Test getting devices with explicit count (falls back to CPU when no GPU)."""
        config = {"num_gpus_per_service": 4}

        devices = get_gpus_for_node(config, node_rank=0)

        # Without actual CUDA, falls back to CPU devices
        assert len(devices) == 4
        # Devices should be either CUDA (if available) or CPU
        assert all(d.startswith("cuda:") or d == "cpu" for d in devices)

    def test_get_devices_single(self):
        """Test getting single device."""
        config = {"num_gpus_per_service": 1}

        devices = get_gpus_for_node(config, node_rank=0)

        assert len(devices) == 1
        assert devices[0].startswith("cuda:") or devices[0] == "cpu"

    def test_get_devices_for_different_ranks(self):
        """Test getting devices for different node ranks."""
        config = {"num_gpus_per_service": 2}

        devices_0 = get_gpus_for_node(config, node_rank=0)
        devices_1 = get_gpus_for_node(config, node_rank=1)

        # With explicit count, all ranks get the same devices
        assert devices_0 == devices_1
        assert len(devices_0) == 2
