"""Unit tests for read-only sandbox tools and path security (M2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from boundary.models import ToolCall
from boundary.tools import ToolSandbox


@pytest.fixture
def sample_sandbox(tmp_path: Path) -> ToolSandbox:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "z_file.py").write_text("alpha = 1\nbeta = 2\n", encoding="utf-8")
    sub = root / "pkg"
    sub.mkdir()
    (sub / "a_module.py").write_text(
        "line_one\nline_two_target\nline_three\nline_four\n",
        encoding="utf-8",
    )
    (sub / "b_module.py").write_text("target_value = 42\n", encoding="utf-8")
    # Create an outside file to test traversal attacks
    (tmp_path / "outside_secret.txt").write_text("TOP_SECRET_KEY", encoding="utf-8")
    return ToolSandbox(root)


def test_list_files_deterministic_sorted(sample_sandbox: ToolSandbox) -> None:
    out = sample_sandbox.list_files(".")
    assert out.splitlines() == [
        "pkg/a_module.py",
        "pkg/b_module.py",
        "z_file.py",
    ]


def test_read_file_full_and_line_ranges(sample_sandbox: ToolSandbox) -> None:
    full = sample_sandbox.read_file("z_file.py")
    assert full == "alpha = 1\nbeta = 2\n"

    sliced = sample_sandbox.read_file("pkg/a_module.py", start_line=2, end_line=3)
    assert sliced == "2: line_two_target\n3: line_three"

    # Invalid ranges return readable error text without raising exceptions
    bad_range = sample_sandbox.read_file("pkg/a_module.py", start_line=4, end_line=2)
    assert bad_range.startswith("Error:")

    zero_line = sample_sandbox.read_file("pkg/a_module.py", start_line=0)
    assert zero_line.startswith("Error:")

    missing = sample_sandbox.read_file("does_not_exist.py")
    assert missing.startswith("Error:")


@pytest.mark.parametrize(
    "malicious_path",
    [
        "../outside_secret.txt",
        "..\\outside_secret.txt",
        "pkg/../../outside_secret.txt",
        "pkg\\..\\..\\outside_secret.txt",
        "C:foo.py",
        "c:..\\outside_secret.txt",
        "C:\\Windows\\win.ini",
        "/etc/passwd",
        "\\Windows\\win.ini",
        "\\\\?\\C:\\outside_secret.txt",
        "\\\\server\\share\\secret.txt",
        "z_file.py:Zone.Identifier",
        "pkg/a_module.py:secret_stream",
    ],
)
def test_path_traversal_and_windows_edge_cases_rejected(
    sample_sandbox: ToolSandbox, malicious_path: str
) -> None:
    for out in (
        sample_sandbox.list_files(malicious_path),
        sample_sandbox.read_file(malicious_path),
        sample_sandbox.grep("SECRET", path=malicious_path),
    ):
        assert out.startswith("Error: access denied")
        assert "TOP_SECRET_KEY" not in out


def test_symlink_escape_rejected(sample_sandbox: ToolSandbox, tmp_path: Path) -> None:
    outside = tmp_path / "outside_secret.txt"
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "leaked.py").write_text("LEAKED_DATA", encoding="utf-8")

    link_file = sample_sandbox.root_dir / "evil_link.txt"
    link_dir = sample_sandbox.root_dir / "evil_dir"
    try:
        link_file.symlink_to(outside)
        link_dir.symlink_to(outside_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation not permitted in this Windows environment: {exc}")

    read_out = sample_sandbox.read_file("evil_link.txt")
    assert read_out.startswith("Error: access denied")
    assert "TOP_SECRET_KEY" not in read_out

    listed = sample_sandbox.list_files(".")
    assert "evil_link.txt" not in listed
    assert "evil_dir/leaked.py" not in listed

    grep_out = sample_sandbox.grep("LEAKED_DATA|TOP_SECRET_KEY")
    assert grep_out == "No matches found."


def test_grep_deterministic_and_bounds(tmp_path: Path) -> None:
    root = tmp_path / "grep_repo"
    root.mkdir()
    (root / "b.py").write_text("match_here_b\n", encoding="utf-8")
    (root / "a.py").write_text(
        "match_here_a1\n" + ("x" * 100 + "match_here_long\n") + "match_here_a3\n",
        encoding="utf-8",
    )
    # Binary file containing pattern bytes after null byte
    (root / "binary.bin").write_bytes(b"\x00\x01\x02match_here_binary\n")
    # Oversized file exceeding grep_max_file_bytes
    (root / "huge.py").write_text("match_here_huge\n" + ("padding\n" * 500), encoding="utf-8")

    sandbox = ToolSandbox(
        root,
        grep_max_matches=3,
        grep_max_line_chars=50,
        grep_max_file_bytes=500,
    )

    # 1. Invalid regex returns error string without raising
    bad_regex = sandbox.grep("[unclosed_group")
    assert bad_regex.startswith("Error: invalid regular expression")

    # 2. Binary file and huge file are skipped; long line is capped at grep_max_line_chars
    res = sandbox.grep("match_here")
    lines = res.splitlines()
    # In a.py: line 1 matches, line 2 has 'match_here_long' past char 50 so not matched,
    # line 3 matches. In b.py: line 1 matches.
    assert lines == [
        "a.py:1:match_here_a1",
        "a.py:3:match_here_a3",
        "b.py:1:match_here_b",
        "[max_matches=3 reached]",
    ]

    # 3. Line truncation when match occurs within grep_max_line_chars on a long line
    (root / "c.py").write_text("early_token_" + ("z" * 100) + "\n", encoding="utf-8")
    long_line_res = sandbox.grep("early_token_", path="c.py")
    assert long_line_res.startswith("c.py:1:early_token_")
    assert long_line_res.endswith("...[line truncated]")


def test_output_truncation_and_execute_tool_result(tmp_path: Path) -> None:
    root = tmp_path / "trunc_repo"
    root.mkdir()
    content = "A" * 500
    (root / "big.txt").write_text(content, encoding="utf-8")

    sandbox = ToolSandbox(root, max_output_chars=120)
    call = ToolCall(id="call_99", name="read_file", arguments={"path": "big.txt"})
    result = sandbox.execute(call)

    assert result.call_id == "call_99"
    assert result.content.endswith("\n[truncated]")
    assert result.content.startswith("A" * 120)
    assert result.size_chars == len(result.content)
    assert result.tokens_est == len(result.content) // 4

    # Unknown tool returns Error string rather than raising
    unknown = sandbox.execute(ToolCall(id="call_u", name="shell_exec", arguments={}))
    assert unknown.content == "Error: unknown tool 'shell_exec'."
