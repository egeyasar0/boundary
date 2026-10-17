"""Read-only sandboxed tools for exploring a fixture repository (R7)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from boundary.models import ToolCall, ToolResult

DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_GREP_MAX_MATCHES = 200
DEFAULT_GREP_MAX_LINE_CHARS = 500
DEFAULT_GREP_MAX_FILE_BYTES = 1_000_000
TRUNCATED_SUFFIX = "\n[truncated]"

TOOL_NAMES: list[str] = ["list_files", "read_file", "grep"]

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a relative directory path inside the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path (default '.').",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file (optionally bounded by 1-indexed start_line and end_line)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed inclusive start line.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed inclusive end line.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a regular expression pattern across files in the repository."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional relative file or directory path to restrict search."
                        ),
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
]


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars // 4 heuristic."""
    return len(text) // 4


def _is_binary_bytes(sample: bytes) -> bool:
    """Return True if byte sample contains a NUL byte indicating binary content."""
    return b"\x00" in sample


class ToolSandbox:
    """Read-only filesystem sandbox locked to root_dir."""

    def __init__(
        self,
        root_dir: Path | str,
        *,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        grep_max_matches: int = DEFAULT_GREP_MAX_MATCHES,
        grep_max_line_chars: int = DEFAULT_GREP_MAX_LINE_CHARS,
        grep_max_file_bytes: int = DEFAULT_GREP_MAX_FILE_BYTES,
    ) -> None:
        self.root_dir: Path = Path(root_dir).resolve()
        self._norm_root: str = os.path.normcase(str(self.root_dir))
        self.max_output_chars = max_output_chars
        self.grep_max_matches = grep_max_matches
        self.grep_max_line_chars = grep_max_line_chars
        self.grep_max_file_bytes = grep_max_file_bytes

    def _resolve_safe_path(self, raw_path: str | None) -> tuple[Path | None, str | None]:
        """Validate and resolve a user-supplied relative path within root_dir.

        Returns (resolved_path, None) on success or (None, error_message) on rejection.
        """
        if raw_path is None or raw_path == "":
            raw_path = "."

        if not isinstance(raw_path, str):
            return None, "Error: path must be a string."

        if "\x00" in raw_path:
            return None, "Error: path contains invalid null byte."

        # Reject Windows drive letters (C:\foo, C:foo), UNC/device prefixes (\\?\, \\server),
        # and NTFS Alternate Data Streams (file.txt:stream).
        if ":" in raw_path:
            return (
                None,
                f"Error: access denied for path '{raw_path}' "
                "(drive prefixes and alternate data streams are forbidden).",
            )

        if raw_path.startswith(("/", "\\")):
            return (
                None,
                f"Error: access denied for path '{raw_path}' "
                "(absolute or UNC paths are forbidden).",
            )

        normalized_slashes = raw_path.replace("\\", "/")
        parts = [p for p in normalized_slashes.split("/") if p not in ("", ".")]
        if any(part == ".." for part in parts):
            return (
                None,
                f"Error: access denied for path '{raw_path}' "
                "(parent directory traversal '..' is forbidden).",
            )

        try:
            candidate = (self.root_dir / Path(*parts)).resolve()
        except OSError as exc:
            return None, f"Error: invalid path '{raw_path}': {exc}"

        if not self._is_within_root(candidate):
            return (
                None,
                f"Error: access denied for path '{raw_path}' "
                "(resolved path escapes sandbox root).",
            )

        return candidate, None

    def _is_within_root(self, candidate: Path) -> bool:
        """Check whether resolved candidate path stays inside root_dir (case-insensitive safe)."""
        try:
            resolved = candidate.resolve()
        except OSError:
            return False

        norm_cand = os.path.normcase(str(resolved))
        if norm_cand == self._norm_root:
            return True
        prefix = self._norm_root.rstrip(os.sep) + os.sep
        return norm_cand.startswith(prefix)

    def _rel_posix(self, path: Path) -> str:
        """Return deterministic POSIX path relative to root_dir."""
        return path.resolve().relative_to(self.root_dir).as_posix()

    def _finalize_output(self, content: str) -> str:
        """Apply output character limit and append [truncated] if cut."""
        if len(content) > self.max_output_chars:
            return content[: self.max_output_chars] + TRUNCATED_SUFFIX
        return content

    def list_files(self, path: str = ".") -> str:
        """List all regular files under path in deterministic sorted order."""
        target, err = self._resolve_safe_path(path)
        if err is not None or target is None:
            return err or "Error: invalid path."

        if not target.exists():
            return f"Error: path '{path}' does not exist."

        if target.is_file():
            return self._finalize_output(self._rel_posix(target))

        collected: list[str] = []
        try:
            for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
                dir_path = Path(dirpath)
                # Filter out symlinked directories or directories outside root
                safe_dirs: list[str] = []
                for d in sorted(dirnames):
                    d_full = dir_path / d
                    if not d_full.is_symlink() and self._is_within_root(d_full):
                        safe_dirs.append(d)
                dirnames[:] = safe_dirs

                for fname in sorted(filenames):
                    f_full = dir_path / fname
                    if f_full.is_symlink():
                        if not self._is_within_root(f_full):
                            continue
                    if f_full.is_file() and self._is_within_root(f_full):
                        collected.append(self._rel_posix(f_full))
        except OSError as exc:
            return f"Error: failed to list files in '{path}': {exc}"

        collected.sort()
        return self._finalize_output("\n".join(collected))

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read text file content with optional 1-indexed inclusive line range."""
        target, err = self._resolve_safe_path(path)
        if err is not None or target is None:
            return err or "Error: invalid path."

        if not target.exists():
            return f"Error: file '{path}' does not exist."
        if not target.is_file():
            return f"Error: path '{path}' is not a file."

        if start_line is not None and (not isinstance(start_line, int) or start_line < 1):
            return "Error: start_line must be an integer >= 1."
        if end_line is not None and (not isinstance(end_line, int) or end_line < 1):
            return "Error: end_line must be an integer >= 1."
        if start_line is not None and end_line is not None and start_line > end_line:
            return f"Error: start_line ({start_line}) cannot be greater than end_line ({end_line})."

        try:
            raw = target.read_bytes()
        except OSError as exc:
            return f"Error: could not read file '{path}': {exc}"

        if _is_binary_bytes(raw[:8192]):
            return f"Error: file '{path}' appears to be a binary file."

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")

        text = text.replace("\r\n", "\n")

        if start_line is None and end_line is None:
            return self._finalize_output(text)

        lines = text.splitlines()
        start_idx = (start_line - 1) if start_line is not None else 0
        end_idx = end_line if end_line is not None else len(lines)
        selected = lines[start_idx:end_idx]
        numbered = [
            f"{idx}: {line}" for idx, line in enumerate(selected, start=start_idx + 1)
        ]
        return self._finalize_output("\n".join(numbered))

    def grep(self, pattern: str, path: str | None = None) -> str:
        """Search files for regex pattern in deterministic order with safety bounds."""
        if not isinstance(pattern, str) or pattern == "":
            return "Error: pattern must be a non-empty string."

        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regular expression '{pattern}': {exc}"

        target, err = self._resolve_safe_path(path if path is not None else ".")
        if err is not None or target is None:
            return err or "Error: invalid path."

        if not target.exists():
            return f"Error: path '{path}' does not exist."

        files_to_search: list[tuple[str, Path]] = []
        if target.is_file():
            files_to_search.append((self._rel_posix(target), target))
        else:
            for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
                dir_path = Path(dirpath)
                safe_dirs: list[str] = []
                for d in sorted(dirnames):
                    d_full = dir_path / d
                    if not d_full.is_symlink() and self._is_within_root(d_full):
                        safe_dirs.append(d)
                dirnames[:] = safe_dirs

                for fname in sorted(filenames):
                    f_full = dir_path / fname
                    if f_full.is_symlink() and not self._is_within_root(f_full):
                        continue
                    if f_full.is_file() and self._is_within_root(f_full):
                        files_to_search.append((self._rel_posix(f_full), f_full))

        files_to_search.sort(key=lambda item: item[0])

        matches: list[str] = []
        match_count = 0
        limit_reached = False

        for rel_path, file_path in files_to_search:
            try:
                stat = file_path.stat()
                if stat.st_size > self.grep_max_file_bytes:
                    continue
                raw = file_path.read_bytes()
            except OSError:
                continue

            if _is_binary_bytes(raw[:8192]):
                continue

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                scan_line = line[: self.grep_max_line_chars]
                if compiled.search(scan_line):
                    display_line = scan_line
                    if len(line) > self.grep_max_line_chars:
                        display_line += "...[line truncated]"
                    matches.append(f"{rel_path}:{line_no}:{display_line}")
                    match_count += 1
                    if match_count >= self.grep_max_matches:
                        limit_reached = True
                        break
            if limit_reached:
                break

        if not matches:
            return "No matches found."

        output = "\n".join(matches)
        if limit_reached:
            output += f"\n[max_matches={self.grep_max_matches} reached]"
        return self._finalize_output(output)

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a ToolCall safely without raising user-input exceptions."""
        args = tool_call.arguments
        name = tool_call.name

        if name == "list_files":
            raw_path = args.get("path", ".")
            content = self.list_files(path=raw_path)
        elif name == "read_file":
            if "path" not in args:
                content = "Error: missing required argument 'path' for read_file."
            else:
                content = self.read_file(
                    path=args["path"],
                    start_line=args.get("start_line"),
                    end_line=args.get("end_line"),
                )
        elif name == "grep":
            if "pattern" not in args:
                content = "Error: missing required argument 'pattern' for grep."
            else:
                content = self.grep(
                    pattern=args["pattern"],
                    path=args.get("path"),
                )
        else:
            content = f"Error: unknown tool '{name}'."

        return ToolResult(
            call_id=tool_call.id,
            content=content,
            size_chars=len(content),
            tokens_est=estimate_tokens(content),
        )
