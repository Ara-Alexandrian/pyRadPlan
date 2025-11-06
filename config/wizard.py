"""Configuration wizard for pyRadPlan dashboard.

Provides interactive setup for dynamic configuration across different environments
(work/home, different networks, different Unraid servers).
"""

import os
import sys
import socket
import json
import yaml
import secrets
from pathlib import Path
from typing import Dict, Any, Optional
import questionary
from questionary import Style


# Styling for interactive prompts
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#f44336 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#cc5454'),
    ('separator', 'fg:#cc5454'),
    ('instruction', ''),
    ('text', ''),
    ('disabled', 'fg:#858585 italic')
])


class ConfigWizard:
    """Interactive configuration wizard for pyRadPlan dashboard."""

    def __init__(self):
        self.config_dir = Path(__file__).parent
        self.repo_root = self.config_dir.parent
        self.templates_dir = self.config_dir / "templates"
        self.config: Dict[str, Any] = {}

    def detect_environment(self) -> Dict[str, Any]:
        """Auto-detect current environment details."""
        hostname = socket.gethostname()

        # Try to get local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"

        # Check if running in container
        in_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')

        return {
            "hostname": hostname,
            "local_ip": local_ip,
            "in_container": in_container,
            "cwd": str(Path.cwd()),
        }

    def validate_path(self, path_str: str, create_if_missing: bool = True) -> bool:
        """Validate a file system path."""
        path = Path(path_str).expanduser()

        if path.exists():
            return path.is_dir()
        elif create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created directory: {path}")
                return True
            except Exception as e:
                print(f"✗ Failed to create {path}: {e}")
                return False
        return False

    def test_port(self, host: str, port: int) -> bool:
        """Test if a port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result != 0  # Port is available if connection fails
        except Exception:
            return False

    def generate_secret(self, length: int = 32) -> str:
        """Generate a random secret key."""
        return secrets.token_urlsafe(length)

    def setup_interactive(self) -> Dict[str, Any]:
        """Run interactive setup wizard."""
        print("\n" + "=" * 60)
        print("  pyRadPlan Dashboard Configuration Wizard")
        print("=" * 60 + "\n")

        env_info = self.detect_environment()
        print("Detected environment:")
        print(f"  Hostname: {env_info['hostname']}")
        print(f"  Local IP: {env_info['local_ip']}")
        print(f"  Container: {'Yes' if env_info['in_container'] else 'No'}")
        print(f"  Working directory: {env_info['cwd']}\n")

        # Ask for environment name
        env_name = questionary.text(
            "Environment name (e.g., 'work', 'home', 'lab'):",
            default="work",
            style=custom_style
        ).ask()

        env_location = questionary.text(
            "Environment description (e.g., 'Unraid Work Server'):",
            default=f"{env_name.capitalize()} Server",
            style=custom_style
        ).ask()

        # Network configuration
        print("\n--- Network Configuration ---\n")

        backend_host = questionary.text(
            "Backend host (0.0.0.0 for all interfaces):",
            default="0.0.0.0",
            style=custom_style
        ).ask()

        backend_port = questionary.text(
            "Backend port:",
            default="8000",
            style=custom_style,
            validate=lambda x: x.isdigit() and 1 <= int(x) <= 65535
        ).ask()
        backend_port = int(backend_port)

        frontend_host = questionary.text(
            "Frontend host (0.0.0.0 for all interfaces):",
            default="0.0.0.0",
            style=custom_style
        ).ask()

        frontend_port = questionary.text(
            "Frontend port:",
            default="5173",
            style=custom_style,
            validate=lambda x: x.isdigit() and 1 <= int(x) <= 65535
        ).ask()
        frontend_port = int(frontend_port)

        # Auto-detect or manual base URL
        use_auto_url = questionary.confirm(
            f"Use auto-detected base URL (http://{env_info['local_ip']})?",
            default=True,
            style=custom_style
        ).ask()

        if use_auto_url:
            base_url = f"http://{env_info['local_ip']}"
        else:
            base_url = questionary.text(
                "Base URL (e.g., http://192.168.1.100):",
                default=f"http://{env_info['local_ip']}",
                style=custom_style
            ).ask()

        # Path configuration
        print("\n--- Path Configuration ---\n")

        # Patient data path
        default_patient_path = "/config/workspace/data/patients"
        patient_data_path = questionary.text(
            "Patient data directory:",
            default=default_patient_path,
            style=custom_style
        ).ask()

        # Results path
        default_results_path = "/config/workspace/data/results"
        results_path = questionary.text(
            "Results directory:",
            default=default_results_path,
            style=custom_style
        ).ask()

        # Temp path
        temp_path = questionary.text(
            "Temporary files directory:",
            default="/tmp/pyradplan",
            style=custom_style
        ).ask()

        # Validate and create paths
        print("\nValidating paths...")
        self.validate_path(patient_data_path)
        self.validate_path(results_path)
        self.validate_path(temp_path)

        # Database configuration
        print("\n--- Database Configuration ---\n")

        db_type = questionary.select(
            "Database type:",
            choices=["sqlite", "postgresql"],
            default="sqlite",
            style=custom_style
        ).ask()

        if db_type == "sqlite":
            db_config = {
                "type": "sqlite",
                "path": "/config/workspace/data/pyradplan.db"
            }
        else:
            db_config = {
                "type": "postgresql",
                "host": questionary.text("PostgreSQL host:", default="postgres", style=custom_style).ask(),
                "port": int(questionary.text("PostgreSQL port:", default="5432", style=custom_style).ask()),
                "database": questionary.text("Database name:", default="pyradplan", style=custom_style).ask(),
                "user": questionary.text("Database user:", default="pyradplan", style=custom_style).ask(),
                "password_env": "DB_PASSWORD"
            }

        # Job queue configuration
        print("\n--- Job Queue Configuration ---\n")

        use_redis = questionary.confirm(
            "Use Redis for job queue (recommended for production)?",
            default=True,
            style=custom_style
        ).ask()

        if use_redis:
            redis_url = questionary.text(
                "Redis URL:",
                default="redis://localhost:6379/0",
                style=custom_style
            ).ask()

            max_workers = questionary.text(
                "Maximum worker processes:",
                default="4",
                style=custom_style,
                validate=lambda x: x.isdigit() and 1 <= int(x) <= 16
            ).ask()

            jobs_config = {
                "queue": "redis",
                "redis_url": redis_url,
                "max_workers": int(max_workers)
            }
        else:
            jobs_config = {
                "queue": "memory",
                "max_workers": 2
            }

        # Authentication
        print("\n--- Authentication ---\n")

        enable_auth = questionary.confirm(
            "Enable authentication (JWT)?",
            default=False,
            style=custom_style
        ).ask()

        auth_config = {
            "enabled": enable_auth
        }

        if enable_auth:
            auth_config["jwt_secret_env"] = "JWT_SECRET"
            auth_config["token_expire_hours"] = 24

        # Build complete config
        config = {
            "environment": {
                "name": env_name,
                "location": env_location
            },
            "network": {
                "backend_host": backend_host,
                "backend_port": backend_port,
                "frontend_host": frontend_host,
                "frontend_port": frontend_port,
                "base_url": base_url
            },
            "paths": {
                "patient_data": patient_data_path,
                "results": results_path,
                "temp": temp_path,
                "machines": "${REPO_ROOT}/pyRadPlan/data/machines"
            },
            "database": db_config,
            "jobs": jobs_config,
            "auth": auth_config
        }

        return config

    def save_config(self, config: Dict[str, Any], env_name: str):
        """Save configuration to YAML file."""
        config_file = self.config_dir / f"{env_name}.yaml"

        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n✓ Configuration saved to: {config_file}")

        # Create symlink to active config
        active_link = self.config_dir / "active.yaml"
        if active_link.exists() or active_link.is_symlink():
            active_link.unlink()
        active_link.symlink_to(config_file.name)

        print(f"✓ Active configuration linked to: {env_name}.yaml")

    def create_env_file(self, config: Dict[str, Any], env_name: str):
        """Create .env file with secrets."""
        env_file = self.repo_root / ".env"

        env_vars = [
            "# pyRadPlan Dashboard Environment Variables",
            f"# Generated for environment: {env_name}",
            "# DO NOT COMMIT THIS FILE TO GIT",
            "",
            f"ENVIRONMENT={env_name}",
            f"BASE_URL={config['network']['base_url']}",
            f"BACKEND_PORT={config['network']['backend_port']}",
            f"FRONTEND_PORT={config['network']['frontend_port']}",
            f"PUBLIC_PORT=80",
            "",
            f"PATIENT_DATA_PATH={config['paths']['patient_data']}",
            f"RESULTS_PATH={config['paths']['results']}",
            f"TEMP_PATH={config['paths']['temp']}",
            "",
        ]

        # Add database password if PostgreSQL
        if config['database']['type'] == 'postgresql':
            db_password = self.generate_secret(16)
            env_vars.extend([
                f"DB_PASSWORD={db_password}",
                ""
            ])

        # Add JWT secret if auth enabled
        if config['auth']['enabled']:
            jwt_secret = self.generate_secret(32)
            env_vars.extend([
                f"JWT_SECRET={jwt_secret}",
                ""
            ])

        # Add Redis URL if using Redis
        if config['jobs'].get('queue') == 'redis':
            env_vars.extend([
                f"REDIS_URL={config['jobs']['redis_url']}",
                ""
            ])

        with open(env_file, 'w') as f:
            f.write('\n'.join(env_vars))

        print(f"✓ Environment file created: {env_file}")
        print("  (This file contains secrets and is git-ignored)")

    def list_environments(self):
        """List all saved environment configurations."""
        configs = list(self.config_dir.glob("*.yaml"))
        configs = [c for c in configs if c.name != "active.yaml"]

        if not configs:
            print("No saved environments found.")
            return

        print("\nSaved environments:")
        for config_file in configs:
            is_active = False
            active_link = self.config_dir / "active.yaml"
            if active_link.is_symlink():
                is_active = active_link.resolve() == config_file

            marker = " (active)" if is_active else ""
            print(f"  - {config_file.stem}{marker}")

    def switch_environment(self, env_name: str):
        """Switch to a different environment."""
        config_file = self.config_dir / f"{env_name}.yaml"

        if not config_file.exists():
            print(f"✗ Environment '{env_name}' not found.")
            print("Run 'setup' to create it first.")
            return False

        active_link = self.config_dir / "active.yaml"
        if active_link.exists() or active_link.is_symlink():
            active_link.unlink()
        active_link.symlink_to(config_file.name)

        print(f"✓ Switched to environment: {env_name}")

        # Update .env file
        with open(config_file) as f:
            config = yaml.safe_load(f)
        self.create_env_file(config, env_name)

        return True

    def test_configuration(self):
        """Test current configuration."""
        active_link = self.config_dir / "active.yaml"

        if not active_link.exists():
            print("✗ No active configuration found.")
            print("Run 'setup' first.")
            return False

        with open(active_link) as f:
            config = yaml.safe_load(f)

        print(f"\nTesting configuration: {config['environment']['name']}")
        print("=" * 60)

        # Test paths
        print("\nPath validation:")
        for path_name, path_value in config['paths'].items():
            if "${REPO_ROOT}" in path_value:
                path_value = path_value.replace("${REPO_ROOT}", str(self.repo_root))
            path = Path(path_value)
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {path_name}: {path_value}")

        # Test ports
        print("\nPort availability:")
        backend_port = config['network']['backend_port']
        frontend_port = config['network']['frontend_port']

        backend_available = self.test_port('127.0.0.1', backend_port)
        frontend_available = self.test_port('127.0.0.1', frontend_port)

        print(f"  {'✓' if backend_available else '✗'} Backend port {backend_port}: {'Available' if backend_available else 'In use'}")
        print(f"  {'✓' if frontend_available else '✗'} Frontend port {frontend_port}: {'Available' if frontend_available else 'In use'}")

        # Test database
        print("\nDatabase:")
        print(f"  Type: {config['database']['type']}")
        if config['database']['type'] == 'sqlite':
            db_path = Path(config['database']['path'])
            print(f"  Path: {db_path}")
            print(f"  Exists: {'Yes' if db_path.exists() else 'No (will be created)'}")

        # Test job queue
        print("\nJob queue:")
        print(f"  Type: {config['jobs']['queue']}")
        if config['jobs']['queue'] == 'redis':
            print(f"  Redis URL: {config['jobs']['redis_url']}")

        print("\n" + "=" * 60)
        return True


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="pyRadPlan Dashboard Configuration Wizard")
    parser.add_argument(
        'command',
        choices=['setup', 'list', 'switch', 'test'],
        help="Command to run"
    )
    parser.add_argument(
        '--environment',
        '-e',
        help="Environment name (for setup/switch commands)"
    )

    args = parser.parse_args()
    wizard = ConfigWizard()

    if args.command == 'setup':
        env_name = args.environment
        if not env_name:
            print("Please provide an environment name with --environment")
            sys.exit(1)

        config = wizard.setup_interactive()
        wizard.save_config(config, env_name)
        wizard.create_env_file(config, env_name)

        print("\n✓ Setup complete!")
        print(f"\nYou can now start the dashboard with:")
        print(f"  docker-compose up -d")

    elif args.command == 'list':
        wizard.list_environments()

    elif args.command == 'switch':
        env_name = args.environment
        if not env_name:
            print("Please provide an environment name with --environment")
            sys.exit(1)
        wizard.switch_environment(env_name)

    elif args.command == 'test':
        wizard.test_configuration()


if __name__ == "__main__":
    main()
