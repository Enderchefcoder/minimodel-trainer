"""Tests for :mod:`minimodel.core`."""

from __future__ import annotations

import json
import logging
import math
import os

import numpy as np
import pytest
import torch

from minimodel.core.config import (
    Config,
    ConfigError,
    load_config,
    merge_configs,
    parse_overrides,
    parse_scalar,
)
from minimodel.core.devices import (
    autocast_context,
    count_parameters,
    describe_device,
    device_memory_stats,
    resolve_device,
    resolve_dtype,
    supports_bf16,
)
from minimodel.core.distributed import (
    DistInfo,
    all_reduce_mean,
    barrier,
    gather_object,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
    teardown_distributed,
)
from minimodel.core.io_utils import (
    append_jsonl,
    atomic_write_text,
    ensure_dir,
    human_bytes,
    human_count,
    human_duration,
    jsonable,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from minimodel.core.logging_utils import JsonlLogger, MetricFormatter, get_logger, setup_logging
from minimodel.core.registry import Registry
from minimodel.core.seeding import (
    get_rng_state,
    seed_worker,
    set_deterministic,
    set_rng_state,
    set_seed,
)


class TestConfig:
    """Config loading, merging and dotted access."""

    def test_dotted_get_and_set(self):
        cfg = Config({"model": {"dim": 128, "layers": 4}})
        assert cfg["model.dim"] == 128
        cfg["model.dim"] = 256
        assert cfg["model.dim"] == 256
        cfg.set("training.lr", 1e-3)
        assert cfg["training.lr"] == 1e-3

    def test_missing_key_behaviour(self):
        cfg = Config({"a": 1})
        assert cfg.get("missing") is None
        assert cfg.get("missing", 7) == 7
        with pytest.raises(KeyError):
            _ = cfg["missing"]
        with pytest.raises(ConfigError, match="required config key"):
            cfg.require("missing.nested")

    def test_delete_and_len_and_iter(self):
        cfg = Config({"a": {"b": 1}, "c": 2})
        assert len(cfg) == 2
        assert set(iter(cfg)) == {"a", "c"}
        del cfg["a.b"]
        assert cfg.get("a") == {}
        with pytest.raises(KeyError):
            del cfg["nope"]

    def test_section_returns_config(self):
        cfg = Config({"training": {"lr": 1e-3}})
        section = cfg.section("training")
        assert isinstance(section, Config)
        assert section["lr"] == 1e-3
        assert dict(cfg.section("absent")) == {}
        with pytest.raises(ConfigError, match="not a mapping"):
            Config({"x": 5}).section("x")

    def test_merge_is_recursive_and_pure(self):
        base = {"a": {"b": 1, "c": 2}}
        merged = merge_configs(base, {"a": {"b": 9}, "d": 3})
        assert merged == {"a": {"b": 9, "c": 2}, "d": 3}
        assert base == {"a": {"b": 1, "c": 2}}

    def test_parse_overrides_and_scalars(self):
        assert parse_overrides(["training.lr=3e-4"]) == {"training": {"lr": 0.0003}}
        assert parse_scalar("true") is True
        assert parse_scalar("[1, 2]") == [1, 2]
        assert parse_scalar("hello") == "hello"
        assert parse_scalar("") == ""
        with pytest.raises(ConfigError):
            parse_overrides(["novalue"])
        with pytest.raises(ConfigError):
            parse_overrides(["=v"])

    def test_flatten_yaml_json_and_save(self, tmp_path):
        cfg = Config({"a": {"b": 1}, "c": [1, 2]})
        assert cfg.flatten() == {"a.b": 1, "c": [1, 2]}
        assert "b: 1" in cfg.to_yaml()
        assert json.loads(cfg.to_json())["a"]["b"] == 1
        path = cfg.save(tmp_path / "out" / "cfg.yaml")
        assert path.exists()

    def test_extends_chain(self, tmp_path):
        (tmp_path / "base.yaml").write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
        (tmp_path / "child.yaml").write_text(
            "extends: base.yaml\nb:\n  c: 9\n  d: 3\n", encoding="utf-8"
        )
        cfg = load_config(tmp_path / "child.yaml")
        assert cfg["a"] == 1
        assert cfg["b.c"] == 9
        assert cfg["b.d"] == 3

    def test_extends_missing_file_raises(self, tmp_path):
        (tmp_path / "c.yaml").write_text("extends: nope.yaml\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "c.yaml")

    def test_env_interpolation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MM_TEST_DIR", "/data/x")
        (tmp_path / "c.yaml").write_text(
            "path: ${MM_TEST_DIR}\nother: ${MM_ABSENT:-fallback}\n", encoding="utf-8"
        )
        cfg = load_config(tmp_path / "c.yaml")
        assert cfg["path"] == "/data/x"
        assert cfg["other"] == "fallback"

    def test_env_interpolation_missing_raises(self, tmp_path):
        os.environ.pop("MM_DEFINITELY_ABSENT", None)
        (tmp_path / "c.yaml").write_text("p: ${MM_DEFINITELY_ABSENT}\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="environment variable"):
            load_config(tmp_path / "c.yaml")

    def test_cli_overrides_applied_last(self, tmp_path):
        (tmp_path / "c.yaml").write_text("training:\n  lr: 1.0\n", encoding="utf-8")
        cfg = load_config(
            tmp_path / "c.yaml", overrides={"training": {"lr": 2.0}}, cli_overrides=["training.lr=3"]
        )
        assert cfg["training.lr"] == 3

    def test_non_mapping_top_level_rejected(self, tmp_path):
        (tmp_path / "c.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_config(tmp_path / "c.yaml")

    def test_from_yaml_string_and_repr(self):
        cfg = Config.from_yaml_string("a: 1", source="inline")
        assert cfg["a"] == 1
        assert "inline" in repr(cfg)
        assert cfg.setdefault_path("b", 5) == 5
        assert cfg.setdefault_path("b", 9) == 5


class TestRegistry:
    """Registry lookup, aliases and errors."""

    def test_register_get_and_alias(self):
        registry: Registry[type] = Registry("thing")

        @registry.register("my-thing", aliases=("mt",))
        class Thing:
            pass

        assert registry.get("my_thing") is Thing
        assert registry.get("MT") is Thing
        assert "my-thing" in registry
        assert registry.names() == ["my_thing"]
        assert registry.aliases() == {"mt": "my_thing"}
        assert len(registry) == 1

    def test_duplicate_registration_rejected(self):
        registry: Registry[int] = Registry("n")
        registry.add("a", 1)
        with pytest.raises(KeyError, match="already contains"):
            registry.add("a", 2)
        registry.add("a", 3, overwrite=True)
        assert registry.get("a") == 3

    def test_unknown_lists_options(self):
        registry: Registry[int] = Registry("n")
        registry.add("alpha", 1)
        with pytest.raises(KeyError, match="alpha"):
            registry.get("beta")
        assert 5 not in registry
        assert "Registry" in repr(registry)

    def test_create_calls_factory(self):
        registry: Registry[type] = Registry("f")
        registry.add("list", list)
        assert registry.create("list") == []


class TestIOUtils:
    """Filesystem helpers and formatting."""

    def test_atomic_write_and_read(self, tmp_path):
        path = atomic_write_text(tmp_path / "a" / "b.txt", "hello")
        assert path.read_text() == "hello"

    def test_json_roundtrip_with_tensors(self, tmp_path):
        payload = {"t": torch.tensor(3.0), "a": np.arange(3), "p": tmp_path}
        write_json(tmp_path / "x.json", payload)
        loaded = read_json(tmp_path / "x.json")
        assert loaded["t"] == 3.0
        assert loaded["a"] == [0, 1, 2]

    def test_jsonl_roundtrip_and_append(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        write_jsonl(path, [{"i": 1}, {"i": 2}])
        append_jsonl(path, {"i": 3})
        assert [r["i"] for r in read_jsonl(path)] == [1, 2, 3]

    def test_jsonl_bad_line_handling(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"a": 1}\nnot json\n\n', encoding="utf-8")
        assert len(list(read_jsonl(path, skip_bad_lines=True))) == 1
        with pytest.raises(ValueError, match="not valid JSON"):
            list(read_jsonl(path))

    def test_jsonable_falls_back_to_str(self):
        class Weird:
            def __repr__(self):
                return "weird"

        assert jsonable(Weird()) == "weird"
        assert jsonable({1: {2}}) == {"1": [2]}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(512, "512 B"), (1536, "1.5 KiB"), (1024**3, "1.0 GiB")],
    )
    def test_human_bytes(self, value, expected):
        assert human_bytes(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(999, "999"), (1_406_240, "1.41M"), (2.5e9, "2.50B"), (3.5, "3.50")],
    )
    def test_human_count(self, value, expected):
        assert human_count(value) == expected

    def test_human_duration(self):
        assert human_duration(12.34) == "12.3s"
        assert human_duration(3661) == "1h 1m 1s"
        assert human_duration(90000).startswith("1d")
        assert human_duration(float("nan")) == "unknown"

    def test_ensure_dir_is_idempotent(self, tmp_path):
        target = tmp_path / "x" / "y"
        assert ensure_dir(target) == ensure_dir(target)


class TestLogging:
    """Console and JSONL logging."""

    def test_get_logger_namespaces(self):
        assert get_logger("minimodel.training.trainer").name == "minimodel.training.trainer"
        assert get_logger().name == "minimodel"
        assert get_logger("custom").name == "minimodel.custom"

    def test_setup_logging_writes_file(self, tmp_path):
        logger = setup_logging(level="DEBUG", log_file=tmp_path / "log" / "run.log", force=True)
        logger.info("hello from the test")
        for handler in logger.handlers:
            handler.flush()
        assert "hello from the test" in (tmp_path / "log" / "run.log").read_text()
        setup_logging(force=True)

    def test_rank_filter_suppresses_non_zero(self, tmp_path):
        logger = setup_logging(rank=1, log_file=tmp_path / "r.log", force=True)
        logger.info("quiet")
        logger.error("loud")
        for handler in logger.handlers:
            handler.flush()
        content = (tmp_path / "r.log").read_text()
        assert "quiet" not in content
        assert "loud" in content
        setup_logging(force=True)

    def test_metric_formatter_orders_and_formats(self):
        text = MetricFormatter().format({"loss": 3.14159, "step": 10, "lr": 0.0003, "z": 1})
        assert text.startswith("step 10 | loss 3.1416 | lr 3.00e-04")
        assert MetricFormatter().format({"x": float("nan")}) == "x nan"
        assert MetricFormatter().format({"flag": True}) == "flag True"

    def test_jsonl_logger_buffers_and_flushes(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        with JsonlLogger(path, flush_every=2) as sink:
            sink.log({"step": 1})
            assert not path.exists()
            sink.log({"step": 2})
            assert path.exists()
        rows = list(read_jsonl(path))
        assert [r["step"] for r in rows] == [1, 2]
        assert "elapsed_s" in rows[0]

    def test_jsonl_logger_disabled(self, tmp_path):
        sink = JsonlLogger(tmp_path / "x.jsonl", enabled=False)
        row = sink.log({"a": 1})
        sink.close()
        assert row["a"] == 1
        assert not (tmp_path / "x.jsonl").exists()

    def test_console_formatter_handles_exceptions(self, caplog):
        logger = get_logger("test-exc")
        with caplog.at_level(logging.ERROR, logger="minimodel.test-exc"):
            try:
                raise ValueError("boom")
            except ValueError:
                logger.exception("failed")
        assert "failed" in caplog.text


class TestSeeding:
    """Deterministic seeding and RNG state."""

    def test_set_seed_is_reproducible(self):
        set_seed(42)
        a = torch.randn(4)
        set_seed(42)
        assert torch.equal(a, torch.randn(4))

    def test_rng_state_roundtrip(self):
        set_seed(7)
        state = get_rng_state()
        first = torch.randn(3)
        set_rng_state(state)
        assert torch.allclose(first, torch.randn(3))

    def test_set_rng_state_tolerates_partial(self):
        set_rng_state(None)
        set_rng_state({"python": list(np.random.get_state()[:1]) and __import__("random").getstate()})

    def test_seed_worker_runs(self):
        seed_worker(3)

    def test_set_deterministic_toggles(self):
        set_deterministic(True)
        assert torch.backends.cudnn.deterministic
        set_deterministic(False)
        assert not torch.backends.cudnn.deterministic


class TestDevices:
    """Device and dtype resolution."""

    def test_resolve_device(self):
        assert resolve_device("cpu").type == "cpu"
        assert resolve_device(torch.device("cpu")).type == "cpu"
        assert resolve_device("auto").type in {"cpu", "cuda", "mps"}

    def test_resolve_dtype(self):
        assert resolve_dtype("bf16") is torch.bfloat16
        assert resolve_dtype("fp32") is torch.float32
        assert resolve_dtype(torch.float16) is torch.float16
        assert resolve_dtype("auto", torch.device("cpu")) is torch.float32
        with pytest.raises(ValueError, match="unknown dtype"):
            resolve_dtype("float8")

    def test_autocast_context_is_noop_for_fp32(self):
        with autocast_context(torch.device("cpu"), torch.float32):
            assert not torch.is_autocast_enabled("cpu")
        with autocast_context(torch.device("cpu"), torch.bfloat16):
            pass

    def test_describe_and_memory(self):
        info = describe_device(torch.device("cpu"))
        assert info["type"] == "cpu"
        assert "torch_version" in info
        assert device_memory_stats(torch.device("cpu"))["allocated_mib"] == 0.0
        assert isinstance(supports_bf16(torch.device("cpu")), bool)

    def test_count_parameters_dedups(self):
        layer = torch.nn.Linear(4, 4)
        assert count_parameters(layer) == 4 * 4 + 4
        for param in layer.parameters():
            param.requires_grad_(False)
        assert count_parameters(layer, trainable_only=True) == 0


class TestDistributed:
    """Single-process behaviour of the distributed helpers."""

    def test_single_process_defaults(self):
        assert not is_distributed()
        assert get_rank() == 0
        assert get_local_rank() == 0
        assert get_world_size() >= 1
        assert is_main_process()
        barrier()
        teardown_distributed()

    def test_setup_returns_single_process_info(self):
        info = setup_distributed()
        assert info.world_size == 1
        assert info.is_main
        assert info.to_dict()["rank"] == 0

    def test_all_reduce_and_gather_are_identity(self):
        assert all_reduce_mean(2.5) == 2.5
        assert all_reduce_mean(torch.tensor(3.0)) == 3.0
        assert gather_object({"a": 1}) == [{"a": 1}]

    def test_dist_info_dataclass(self):
        info = DistInfo(rank=1, world_size=2)
        assert not info.is_main
        assert math.isclose(info.world_size, 2)
