"""Configuration loader for pyRadPlan backend.

Loads configuration from YAML file and environment variables.
Supports dynamic environment switching (work/home).
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class NetworkConfig(BaseModel):
    """Network configuration."""

    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000, ge=1, le=65535)
    frontend_host: str = Field(default="0.0.0.0")
    frontend_port: int = Field(default=5173, ge=1, le=65535)
    base_url: str = Field(default="http://localhost")


class PathConfig(BaseModel):
    """Path configuration."""

    patient_data: Path
    results: Path
    temp: Path
    machines: Path

    @field_validator("patient_data", "results", "temp", "machines", mode="before")
    @classmethod
    def expand_paths(cls, v):
        """Expand environment variables and resolve paths."""
        if isinstance(v, str):
            # Replace ${REPO_ROOT} with actual repository root
            if "${REPO_ROOT}" in v:
                repo_root = Path(__file__).parent.parent.parent
                v = v.replace("${REPO_ROOT}", str(repo_root))
            return Path(v).expanduser().resolve()
        return v


class DatabaseConfig(BaseModel):
    """Database configuration."""

    type: str = Field(pattern="^(sqlite|postgresql)$")
    path: Optional[Path] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password_env: Optional[str] = None

    def get_url(self) -> str:
        """Get database connection URL."""
        if self.type == "sqlite":
            return f"sqlite:///{self.path}"
        elif self.type == "postgresql":
            password = os.getenv(self.password_env, "")
            return f"postgresql://{self.user}:{password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unknown database type: {self.type}")


class JobsConfig(BaseModel):
    """Job queue configuration."""

    queue: str = Field(pattern="^(redis|memory)$")
    redis_url: Optional[str] = None
    max_workers: int = Field(default=4, ge=1, le=16)


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = Field(default=False)
    jwt_secret_env: Optional[str] = "JWT_SECRET"
    token_expire_hours: int = Field(default=24, ge=1)

    def get_jwt_secret(self) -> str:
        """Get JWT secret from environment."""
        if self.enabled and self.jwt_secret_env:
            secret = os.getenv(self.jwt_secret_env)
            if not secret:
                raise ValueError(
                    f"JWT secret not found in environment variable: {self.jwt_secret_env}"
                )
            return secret
        return "insecure-development-secret"


class EnvironmentInfo(BaseModel):
    """Environment information."""

    name: str
    location: str


class FeaturesConfig(BaseModel):
    """Feature flags."""

    dicom_import: bool = True
    matlab_export: bool = True
    realtime_updates: bool = True


class Config(BaseModel):
    """Main configuration model."""

    environment: EnvironmentInfo
    network: NetworkConfig
    paths: PathConfig
    database: DatabaseConfig
    jobs: JobsConfig
    auth: AuthConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> "Config":
        """Load configuration from file."""
        if config_file is None:
            # Try to load from active.yaml symlink
            config_dir = Path(__file__).parent.parent.parent / "config"
            config_file = config_dir / "active.yaml"

            if not config_file.exists():
                raise FileNotFoundError(
                    f"Active configuration not found: {config_file}\n"
                    "Run: python -m config.wizard setup --environment <name>"
                )

        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)


# Global config instance (loaded on first import)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        config_file = os.getenv("CONFIG_FILE")
        if config_file:
            _config = Config.load(Path(config_file))
        else:
            _config = Config.load()
    return _config


def reload_config(config_file: Optional[Path] = None):
    """Reload configuration from file."""
    global _config
    _config = Config.load(config_file)


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully:")
    print(f"  Environment: {config.environment.name}")
    print(f"  Backend: {config.network.base_url}:{config.network.backend_port}")
    print(f"  Database: {config.database.type}")
    print(f"  Job queue: {config.jobs.queue}")
    print(f"  Auth: {'enabled' if config.auth.enabled else 'disabled'}")
