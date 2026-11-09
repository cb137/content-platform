"""Settings for the AI layer, read from environment variables (and api/.env)."""

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Each field reads the environment variable with the same name in capitals,
    e.g. llm_provider <- LLM_PROVIDER. Values in a .env file in the current
    folder are used too; real environment variables win over the file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "LLM_PROVIDER=" (blank) means "use the default"
        extra="ignore",
    )

    # SecretStr hides the value when printed or logged: it shows as '**********'.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    llm_provider: Literal["claude", "openai"] = "claude"
    claude_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-6.1-sol"
