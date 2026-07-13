import os
import re
import subprocess
import importlib
import questionary
import ollama

# Anchored to the project root relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONTEXT = 8192


def _get_model_max_context(model_name: str) -> int:
    """Gets the model's native max context by parsing `ollama show` CLI output.
    Falls back to the Python SDK, then to DEFAULT_CONTEXT if both fail.
    """
    # 1. Preferred: parse `ollama show` CLI output directly (most reliable)
    try:
        result = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True, text=True, timeout=15, check=True
        )
        match = re.search(r"context length\s+(\d+)", result.stdout)
        if match:
            return int(match.group(1))
        print(f"[Megamind] ⚠️ Could not find context length in `ollama show {model_name}` output.")
    except subprocess.CalledProcessError as e:
        print(f"[Megamind] ⚠️ `ollama show {model_name}` failed: {e.stderr.strip()}")
    except FileNotFoundError:
        print("[Megamind] ⚠️ `ollama` CLI not found on PATH.")
    except Exception as e:
        print(f"[Megamind] ⚠️ Unexpected error running `ollama show`: {e}")

    # 2. Fallback: Python SDK
    try:
        info = ollama.show(model_name)
        model_info = info.get('model_info', {})
        for key, value in model_info.items():
            if key.endswith('.context_length'):
                return int(value)
    except Exception as e:
        print(f"[Megamind] ⚠️ Python SDK lookup also failed: {e}")

    print(f"[Megamind] ⚠️ Falling back to default context length of {DEFAULT_CONTEXT} — this may not be accurate.")
    return DEFAULT_CONTEXT


def _update_env_var(lines: list, key: str, new_value: str) -> list:
    """Updates or appends a single KEY=value line in a list of .env lines."""
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={new_value}\n"
            found = True
            break
    if not found:
        lines.append(f"{key}={new_value}\n")
    return lines


def _update_config_var(content: str, var_name: str, env_key: str, new_value, is_string: bool) -> str:
    """Updates a `VAR = os.getenv("ENV_KEY", default)` line in config.py source,
    regardless of what the current default value is. Appends a fresh line if
    no existing declaration is found.
    """
    if is_string:
        pattern = rf'{var_name}\s*=\s*os\.getenv\(\s*"{env_key}"\s*,\s*"[^"]*"\s*\)'
        replacement = f'{var_name}      = os.getenv("{env_key}", "{new_value}")'
        fallback_line = f'\n{var_name}      = os.getenv("{env_key}", "{new_value}")\n'
    else:
        pattern = rf'{var_name}\s*=\s*os\.getenv\(\s*"{env_key}"\s*,\s*[0-9]+\s*\)'
        replacement = f'{var_name}   = os.getenv("{env_key}", {new_value})'
        fallback_line = f'\n{var_name}   = os.getenv("{env_key}", {new_value})\n'

    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
    else:
        content += fallback_line
    return content


def _update_config_files(var_name: str, env_key: str, new_model: str, new_context):
    """Safely rewrites <var_name>/<var_name>_CONTEXT-style pairs in both .env
    and core/config.py, then hot-reloads core.config in the current process.

    var_name/env_key examples:
        ("CHAT_MODEL", "CHAT_MODEL") + ("CONTEXT_LIMIT", "CONTEXT_LIMIT")
        ("ROUTER_MODEL", "ROUTER_MODEL") + ("ROUTER_CONTEXT", "ROUTER_CONTEXT")
    """
    context_var = "CONTEXT_LIMIT" if var_name == "CHAT_MODEL" else "ROUTER_CONTEXT"

    # 1. Update .env file
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        lines = _update_env_var(lines, env_key, new_model)
        lines = _update_env_var(lines, context_var, str(new_context))

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("[Megamind] ✅ Configured .env file.")

    # 2. Update core/config.py file
    config_path = os.path.join(BASE_DIR, "core", "config.py")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = _update_config_var(content, var_name, env_key, new_model, is_string=True)
        content = _update_config_var(content, context_var, context_var, new_context, is_string=False)

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[Megamind] ✅ Configured core/config.py file.")

    # 3. Hot-reload core.config in the current process (no restart needed)
    try:
        from core import config as core_config
        importlib.reload(core_config)
        print("[Megamind] 🔄 Reloaded core.config in memory — no restart needed.")
    except Exception as e:
        print(f"[Megamind] ⚠️ Could not hot-reload config ({e}). Restart Megamind to apply changes.")


def _run_model_migration(title: str, var_name: str, env_key: str):
    """Shared interactive flow for both CHAT_MODEL and ROUTER_MODEL migrations."""
    print(f"\n=== Megamind {title} Migrator ===")

    try:
        local_models = [m['model'] for m in ollama.list().get('models', [])]
    except Exception:
        local_models = []
        print("[Megamind] ⚠️ Could not connect to Ollama to pull local model list.")

    choices = local_models.copy()
    choices.append("[Enter Custom Model Name]")
    choices.append("Cancel")

    selection = questionary.select(
        f"Select the new {env_key} for Megamind:",
        choices=choices
    ).ask()

    if selection == "Cancel" or selection is None:
        print("[Megamind] Operation cancelled.")
        return

    if selection == "[Enter Custom Model Name]":
        new_model = questionary.text("Type the exact Ollama model string:").ask()
        if not new_model or not new_model.strip():
            return
        new_model = new_model.strip()
    else:
        new_model = selection

    print(f"\n[Megamind] 🔍 Analyzing '{new_model}' metadata...")
    max_context = _get_model_max_context(new_model)

    context_input = questionary.text(
        f"Set context limit (Native max is {max_context}). Press Enter for default ({DEFAULT_CONTEXT}):",
        validate=lambda text: text.isdigit() or text.strip() == "" or "Please enter a valid number."
    ).ask()

    if context_input is None:
        return

    new_context = context_input.strip() if context_input.strip() != "" else str(DEFAULT_CONTEXT)

    print(f"\nTarget Summary:\n - Model: {new_model}\n - Context: {new_context} tokens")
    confirm = questionary.confirm("Apply these changes across configuration files?").ask()

    if confirm:
        print("\n[Megamind] 🔄 Updating configs...")
        _update_config_files(var_name, env_key, new_model, new_context)
        print(f"[Megamind] 🚀 Successfully migrated {title}!\n")
    else:
        print("[Megamind] Cancelled modification.")


def change_chat_model():
    """Interactive tool: change the main CHAT_MODEL + CONTEXT_LIMIT."""
    _run_model_migration("Chat Model", "CHAT_MODEL", "CHAT_MODEL")


def change_router_model():
    """Interactive tool: change the ROUTER_MODEL + ROUTER_CONTEXT."""
    _run_model_migration("Router Model", "ROUTER_MODEL", "ROUTER_MODEL")