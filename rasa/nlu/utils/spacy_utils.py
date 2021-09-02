from __future__ import annotations

from abc import abstractmethod
import dataclasses
from pathlib import Path
import typing
import logging
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
import rasa.utils.train_utils
from rasa.nlu.model import InvalidModelError
from rasa.shared.constants import DOCS_URL_COMPONENTS

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language  # noqa: F401

# TODO: This is a workaround around until we have all components migrated to
# `GraphComponent`.
SpacyNLP = rasa.nlu.utils._spacy_utils.SpacyNLP


@dataclasses.dataclass
class SpacyModel:
    """Wraps `SpacyNLPGraphComponent` output to make it fingerprintable."""

    def __init__(
        self,
        model: "Language",
        model_path: Path,
    ) -> None:
        """Initializing SpacyModel."""
        self.model = model
        self.model_path = model_path

    def fingerprint(self) -> Text:
        """Fingerprints the model path.

        Use a static fingerprint as we assume this only changes if the file path
        changes and want to avoid investigating the model in greater detail for now.

        Returns:
            Fingerprint for model.
        """
        return str(self.model_path)


class SpacyNLPGraphComponent(GraphComponent):
    """Component which provides the common loaded Spacy model to others.

    This is used to avoid loading the Spacy model multiple times. Instead the Spacy
    model is only loaded once and then shared by depending components.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            # when retrieving word vectors, this will decide if the casing
            # of the word is relevant. E.g. `hello` and `Hello` will
            # retrieve the same vector, if set to `False`. For some
            # applications and models it makes sense to differentiate
            # between these two words, therefore setting this to `True`.
            "case_sensitive": False,
        }

    def __init__(
        self,
        model: "Language" = None,
        model_name: Text = None,
    ) -> None:
        """Initializes a `SpacyNLPGraphComponent`."""
        self.model = model
        self.model_name = model_name

    @staticmethod
    def load_model(spacy_model_name: Text) -> "Language":
        """Try loading the model, catching the OSError if missing."""
        import spacy

        if not spacy_model_name:
            raise InvalidModelError(
                f"Missing model configuration for `SpacyNLP` in `config.yml`.\n"
                f"You must pass a model to the `SpacyNLP` component explicitly.\n"
                f"For example:\n"
                f"- name: SpacyNLP\n"
                f"  model: en_core_web_md\n"
                f"More informaton can be found on {DOCS_URL_COMPONENTS}#spacynlp"
            )

        try:
            return spacy.load(spacy_model_name, disable=["parser"])
        except OSError:
            raise InvalidModelError(
                f"Please confirm that {spacy_model_name} is an available spaCy model. "
                f"You need to download one upfront. For example:\n"
                f"python -m spacy download en_core_web_md\n"
                f"More informaton can be found on {DOCS_URL_COMPONENTS}#spacynlp"
            )

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["spacy"]

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> SpacyNLPGraphComponent:
        """Creates component (see parent class for full docstring)."""

        spacy_model_name = config.get("model")

        logger.info(f"Trying to load spacy model with name '{spacy_model_name}'")

        model = cls.load_model(spacy_model_name)

        cls.ensure_proper_language_model(model)
        return cls(model, spacy_model_name)

    @staticmethod
    def ensure_proper_language_model(nlp: Optional["Language"]) -> None:
        """Checks if the spacy language model is properly loaded.

        Raises an exception if the model is invalid."""

        if nlp is None:
            raise Exception(
                "Failed to load spacy language model. "
                "Loading the model returned 'None'."
            )
        if nlp.path is None:
            # Spacy sets the path to `None` if
            # it did not load the model from disk.
            # In this case `nlp` is an unusable stub.
            raise Exception(
                f"Failed to load spacy language model for "
                f"lang '{nlp.lang}'. Make sure you have downloaded the "
                f"correct model (https://spacy.io/docs/usage/)."
                ""
            )
