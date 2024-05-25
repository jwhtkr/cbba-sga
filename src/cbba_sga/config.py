"""Define the config objects for CBBA/SGA."""

import pathlib
import typing
import pydantic
import yaml


ClassT = typing.TypeVar("ClassT")


class ConfigBase(pydantic.BaseModel):
    """Base class for configuration classes. Primarily configures Pydantic."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


def load_config_yaml(
    config_class: typing.Type[ClassT], files: typing.List[pathlib.Path]
) -> ClassT:
    """Load configuration from yaml files."""
    config_dict = {}
    for file in files:
        with file.open() as f:
            config_dict.update(yaml.safe_load(f))
    return config_class(**config_dict)
