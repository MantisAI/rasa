import uuid
from pathlib import Path
from typing import Type, List, Text, Optional, Dict, Any

import dataclasses
import numpy as np
import pytest
from _pytest.tmpdir import TempPathFactory
from rasa.core.constants import POLICY_MAX_HISTORY
from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage

from rasa.shared.core.generator import TrackerWithCachedStates

from rasa.core import training
from rasa.shared.constants import DEFAULT_SENDER_ID
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    Event,
    UserUttered,
    EntitiesAdded,
    SlotSet,
)
from rasa.core.featurizers.single_state_featurizer import (
    SingleStateFeaturizer,
    IntentTokenizerSingleStateFeaturizer,
)
from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    TrackerFeaturizer,
    IntentMaxHistoryTrackerFeaturizer,
)
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.core.policies.policy import (
    SupportedData,
    Policy,
    InvalidPolicyConfig,
    PolicyGraphComponent,
)
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.memoization import (
    AugmentedMemoizationPolicyGraphComponent as AugmentedMemoizationPolicy,
    MemoizationPolicyGraphComponent as MemoizationPolicy,
)

from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogues import TEST_DEFAULT_DIALOGUE
from tests.core.utilities import get_tracker, tracker_from_dialogue


def train_trackers(
    domain: Domain, stories_file: Text, augmentation_factor: int = 20
) -> List[TrackerWithCachedStates]:
    return training.load_data(
        stories_file, domain, augmentation_factor=augmentation_factor
    )


# We are going to use class style testing here since unfortunately pytest
# doesn't support using fixtures as arguments to its own parameterize yet
# (hence, we can't train a policy, declare it as a fixture and use the
# different fixtures of the different policies for the functional tests).
# Therefore, we are going to reverse this and train the policy within a class
# and collect the tests in a base class.
# noinspection PyMethodMayBeStatic
class PolicyTestCollection:
    """Tests every policy needs to fulfill.

    Each policy can declare further tests on its own."""

    @staticmethod
    def _policy_class_to_test() -> Type[PolicyGraphComponent]:
        raise NotImplementedError

    max_history = 3  # this is the amount of history we test on

    @pytest.fixture(scope="class")
    def resource(self,) -> Resource:
        return Resource(uuid.uuid4().hex)

    @pytest.fixture(scope="class")
    def model_storage(self, tmp_path_factory: TempPathFactory) -> ModelStorage:
        return LocalModelStorage(tmp_path_factory.mktemp(uuid.uuid4().hex))

    @pytest.fixture(scope="class")
    def execution_context(self) -> ExecutionContext:
        return ExecutionContext(GraphSchema({}), uuid.uuid4().hex)

    def _config(
        self, config_override: Optional[Dict[Text, Any]] = None
    ) -> Dict[Text, Any]:
        config_override = config_override or {}
        config = self._policy_class_to_test().get_default_config()
        return {**config, **config_override}

    def create_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        config: Optional[Dict[Text, Any]] = None,
    ) -> PolicyGraphComponent:
        return self._policy_class_to_test()(
            config=self._config(config),
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
            featurizer=featurizer,
        )

    @pytest.fixture(scope="class")
    def featurizer(self) -> TrackerFeaturizer:
        featurizer = MaxHistoryTrackerFeaturizer(
            SingleStateFeaturizer(), max_history=self.max_history
        )
        return featurizer

    @pytest.fixture(scope="class")
    def default_domain(self, domain_path: Text) -> Domain:
        return Domain.load(domain_path)

    @pytest.fixture(scope="class")
    def tracker(self, default_domain: Domain) -> DialogueStateTracker:
        return DialogueStateTracker(DEFAULT_SENDER_ID, default_domain.slots)

    @pytest.fixture(scope="class")
    def trained_policy(
        self,
        featurizer: Optional[TrackerFeaturizer],
        stories_path: Text,
        default_domain: Domain,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> PolicyGraphComponent:
        policy = self.create_policy(
            featurizer, model_storage, resource, execution_context
        )
        training_trackers = train_trackers(
            default_domain, stories_path, augmentation_factor=20
        )
        policy.train(training_trackers, default_domain)
        return policy

    def test_featurizer(
        self,
        trained_policy: PolicyGraphComponent,
        resource: Resource,
        model_storage: ModelStorage,
        tmp_path: Path,
        execution_context: ExecutionContext,
    ):
        assert isinstance(trained_policy.featurizer, MaxHistoryTrackerFeaturizer)
        assert trained_policy.featurizer.max_history == self.max_history
        assert isinstance(
            trained_policy.featurizer.state_featurizer, SingleStateFeaturizer
        )

        loaded = trained_policy.__class__.load(
            self._config(trained_policy.config),
            model_storage,
            resource,
            execution_context,
        )

        assert isinstance(loaded.featurizer, MaxHistoryTrackerFeaturizer)
        assert loaded.featurizer.max_history == self.max_history
        assert isinstance(loaded.featurizer.state_featurizer, SingleStateFeaturizer)

    @pytest.mark.parametrize("should_finetune", [False, True])
    def test_persist_and_load(
        self,
        trained_policy: PolicyGraphComponent,
        default_domain: Domain,
        should_finetune: bool,
        stories_path: Text,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ):
        loaded = trained_policy.__class__.load(
            self._config(trained_policy.config),
            model_storage,
            resource,
            dataclasses.replace(execution_context, is_finetuning=should_finetune),
        )

        assert loaded.finetune_mode == should_finetune

        trackers = train_trackers(default_domain, stories_path, augmentation_factor=20)

        for tracker in trackers:
            predicted_probabilities = loaded.predict_action_probabilities(
                tracker, default_domain, RegexInterpreter()
            )
            actual_probabilities = trained_policy.predict_action_probabilities(
                tracker, default_domain, RegexInterpreter()
            )
            assert predicted_probabilities == actual_probabilities

    def test_prediction_on_empty_tracker(
        self, trained_policy: Policy, default_domain: Domain
    ):
        tracker = DialogueStateTracker(DEFAULT_SENDER_ID, default_domain.slots)
        prediction = trained_policy.predict_action_probabilities(
            tracker, default_domain, RegexInterpreter()
        )
        assert not prediction.is_end_to_end_prediction
        assert len(prediction.probabilities) == default_domain.num_actions
        assert max(prediction.probabilities) <= 1.0
        assert min(prediction.probabilities) >= 0.0

    @pytest.mark.filterwarnings(
        "ignore:.*without a trained model present.*:UserWarning"
    )
    def test_persist_and_load_empty_policy(
        self,
        default_domain: Domain,
        default_model_storage: ModelStorage,
        execution_context: ExecutionContext,
    ):
        resource = Resource(uuid.uuid4().hex)
        empty_policy = self.create_policy(
            None, default_model_storage, resource, execution_context,
        )

        empty_policy.train([], default_domain)
        loaded = empty_policy.__class__.load(
            self._config(), default_model_storage, resource, execution_context,
        )

        assert loaded is not None

    @staticmethod
    def _get_next_action(
        policy: PolicyGraphComponent, events: List[Event], domain: Domain
    ) -> Text:
        tracker = get_tracker(events)

        scores = policy.predict_action_probabilities(
            tracker, domain, RegexInterpreter()
        ).probabilities
        index = scores.index(max(scores))
        return domain.action_names_or_texts[index]

    @pytest.mark.parametrize(
        "featurizer_config, tracker_featurizer, state_featurizer",
        [
            (
                [
                    {
                        "name": "MaxHistoryTrackerFeaturizer",
                        POLICY_MAX_HISTORY: 12,
                        "state_featurizer": [],
                    }
                ],
                MaxHistoryTrackerFeaturizer(max_history=12),
                type(None),
            ),
            (
                [{"name": "MaxHistoryTrackerFeaturizer", POLICY_MAX_HISTORY: 12}],
                MaxHistoryTrackerFeaturizer(max_history=12),
                type(None),
            ),
            (
                [
                    {
                        "name": "IntentMaxHistoryTrackerFeaturizer",
                        POLICY_MAX_HISTORY: 12,
                        "state_featurizer": [
                            {"name": "IntentTokenizerSingleStateFeaturizer"}
                        ],
                    }
                ],
                IntentMaxHistoryTrackerFeaturizer(max_history=12),
                IntentTokenizerSingleStateFeaturizer,
            ),
        ],
    )
    def test_different_featurizer_configs(
        self,
        featurizer_config: Optional[Dict[Text, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        tracker_featurizer: MaxHistoryTrackerFeaturizer,
        state_featurizer: Type[SingleStateFeaturizer],
    ):
        featurizer_config_override = (
            {"featurizer": featurizer_config} if featurizer_config else {}
        )
        policy = self.create_policy(
            None,
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
            config=self._config(featurizer_config_override),
        )

        featurizer = policy.featurizer
        assert isinstance(featurizer, tracker_featurizer.__class__)

        if featurizer_config:
            expected_max_history = featurizer_config[0].get(POLICY_MAX_HISTORY)
        else:
            expected_max_history = self._config().get(POLICY_MAX_HISTORY)

        assert featurizer.max_history == expected_max_history

        assert isinstance(featurizer.state_featurizer, state_featurizer)

    @pytest.mark.parametrize(
        "featurizer_config",
        [
            [
                {"name": "MaxHistoryTrackerFeaturizer", POLICY_MAX_HISTORY: 12},
                {"name": "MaxHistoryTrackerFeaturizer", POLICY_MAX_HISTORY: 12},
            ],
            [
                {
                    "name": "IntentMaxHistoryTrackerFeaturizer",
                    POLICY_MAX_HISTORY: 12,
                    "state_featurizer": [
                        {"name": "IntentTokenizerSingleStateFeaturizer"},
                        {"name": "IntentTokenizerSingleStateFeaturizer"},
                    ],
                }
            ],
        ],
    )
    def test_different_invalid_featurizer_configs(
        self,
        trained_policy: PolicyGraphComponent,
        featurizer_config: Optional[Dict[Text, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ):
        with pytest.raises(InvalidPolicyConfig):
            self.create_policy(
                None,
                model_storage=model_storage,
                resource=resource,
                execution_context=execution_context,
                config={"featurizer": featurizer_config},
            )


class TestMemoizationPolicy(PolicyTestCollection):
    @staticmethod
    def _policy_class_to_test() -> Type[PolicyGraphComponent]:
        return MemoizationPolicy

    @pytest.fixture(scope="class")
    def featurizer(self) -> TrackerFeaturizer:
        featurizer = MaxHistoryTrackerFeaturizer(None, max_history=self.max_history)
        return featurizer

    def test_featurizer(
        self,
        trained_policy: PolicyGraphComponent,
        resource: Resource,
        model_storage: ModelStorage,
        tmp_path: Path,
        execution_context: ExecutionContext,
    ):
        assert isinstance(trained_policy.featurizer, MaxHistoryTrackerFeaturizer)
        assert trained_policy.featurizer.state_featurizer is None
        loaded = trained_policy.__class__.load(
            self._config(trained_policy.config),
            model_storage,
            resource,
            execution_context,
        )
        assert isinstance(loaded.featurizer, MaxHistoryTrackerFeaturizer)
        assert loaded.featurizer.state_featurizer is None

    def test_memorise(
        self,
        trained_policy: MemoizationPolicy,
        default_domain: Domain,
        stories_path: Text,
    ):
        trackers = train_trackers(default_domain, stories_path, augmentation_factor=20)
        trained_policy.train(trackers, default_domain)
        lookup_with_augmentation = trained_policy.lookup

        trackers = [
            t for t in trackers if not hasattr(t, "is_augmented") or not t.is_augmented
        ]

        (
            all_states,
            all_actions,
        ) = trained_policy.featurizer.training_states_and_labels(
            trackers, default_domain
        )

        for tracker, states, actions in zip(trackers, all_states, all_actions):
            recalled = trained_policy.recall(states, tracker, default_domain)
            assert recalled == actions[0]

        nums = np.random.randn(default_domain.num_states)
        random_states = [{f: num for f, num in zip(default_domain.input_states, nums)}]
        assert trained_policy._recall_states(random_states) is None

        # compare augmentation for augmentation_factor of 0 and 20:
        trackers_no_augmentation = train_trackers(
            default_domain, stories_path, augmentation_factor=0
        )
        trained_policy.train(trackers_no_augmentation, default_domain)

        lookup_no_augmentation = trained_policy.lookup

        assert lookup_no_augmentation == lookup_with_augmentation

    def test_memorise_with_nlu(
        self, trained_policy: MemoizationPolicy, default_domain: Domain
    ):
        tracker = tracker_from_dialogue(TEST_DEFAULT_DIALOGUE, default_domain)
        states = trained_policy._prediction_states(tracker, default_domain)

        recalled = trained_policy.recall(states, tracker, default_domain)
        assert recalled is not None

    def test_finetune_after_load(
        self,
        trained_policy: MemoizationPolicy,
        resource: Resource,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        default_domain: Domain,
        stories_path: Text,
    ):

        execution_context = dataclasses.replace(execution_context, is_finetuning=True)
        loaded_policy = MemoizationPolicy.load(
            trained_policy.config, model_storage, resource, execution_context
        )

        assert loaded_policy.finetune_mode

        new_story = TrackerWithCachedStates.from_events(
            "channel",
            domain=default_domain,
            slots=default_domain.slots,
            evts=[
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered(intent={"name": "why"}),
                ActionExecuted("utter_channel"),
                ActionExecuted(ACTION_LISTEN_NAME),
            ],
        )
        original_train_data = train_trackers(
            default_domain, stories_path, augmentation_factor=20
        )
        loaded_policy.train(original_train_data + [new_story], default_domain)

        # Get the hash of the tracker state of new story
        new_story_states, _ = loaded_policy.featurizer.training_states_and_labels(
            [new_story], default_domain
        )

        # Feature keys for each new state should be present in the lookup
        for states in new_story_states:
            state_key = loaded_policy._create_feature_key(states)
            assert state_key in loaded_policy.lookup

    @pytest.mark.parametrize(
        "tracker_events_with_action, tracker_events_without_action",
        [
            (
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                ],
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                ],
            ),
            (
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    EntitiesAdded(entities=[{"entity": "name", "value": "Peter"}]),
                    SlotSet("name", "Peter"),
                    ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
                ],
                [
                    ActionExecuted(ACTION_LISTEN_NAME),
                    UserUttered(text="hello", intent={"name": "greet"}),
                    SlotSet("name", "Peter"),
                    EntitiesAdded(entities=[{"entity": "name", "value": "Peter"}]),
                ],
            ),
        ],
    )
    def test_ignore_action_unlikely_intent(
        self,
        trained_policy: MemoizationPolicy,
        default_domain: Domain,
        tracker_events_with_action: List[Event],
        tracker_events_without_action: List[Event],
    ):
        interpreter = RegexInterpreter()
        tracker_with_action = DialogueStateTracker.from_events(
            "test 1", evts=tracker_events_with_action, slots=default_domain.slots
        )
        tracker_without_action = DialogueStateTracker.from_events(
            "test 2", evts=tracker_events_without_action, slots=default_domain.slots
        )
        prediction_with_action = trained_policy.predict_action_probabilities(
            tracker_with_action, default_domain, interpreter
        )
        prediction_without_action = trained_policy.predict_action_probabilities(
            tracker_without_action, default_domain, interpreter
        )

        # Memoization shouldn't be affected with the
        # presence of action_unlikely_intent.
        assert (
            prediction_with_action.probabilities
            == prediction_without_action.probabilities
        )

    @pytest.mark.parametrize(
        "featurizer_config, tracker_featurizer, state_featurizer",
        [
            (None, MaxHistoryTrackerFeaturizer(), type(None)),
            ([], MaxHistoryTrackerFeaturizer(), type(None)),
        ],
    )
    def test_empty_featurizer_configs(
        self,
        featurizer_config: Optional[Dict[Text, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        tracker_featurizer: MaxHistoryTrackerFeaturizer,
        state_featurizer: Type[SingleStateFeaturizer],
    ):
        featurizer_config_override = (
            {"featurizer": featurizer_config} if featurizer_config else {}
        )
        policy = self.create_policy(
            None,
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
            config=self._config(featurizer_config_override),
        )

        featurizer = policy.featurizer
        assert isinstance(featurizer, tracker_featurizer.__class__)

        if featurizer_config:
            expected_max_history = featurizer_config[0].get(POLICY_MAX_HISTORY)
        else:
            expected_max_history = self._config().get(POLICY_MAX_HISTORY)

        assert featurizer.max_history == expected_max_history

        assert isinstance(featurizer.state_featurizer, state_featurizer)

    @pytest.mark.parametrize("max_history", [1, 2, 3, 4, None])
    def test_prediction(
        self,
        max_history: Optional[int],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ):
        policy = self.create_policy(
            featurizer=MaxHistoryTrackerFeaturizer(max_history=max_history),
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )

        GREET_INTENT_NAME = "greet"
        UTTER_GREET_ACTION = "utter_greet"
        UTTER_BYE_ACTION = "utter_goodbye"
        domain = Domain.from_yaml(
            f"""
            intents:
            - {GREET_INTENT_NAME}
            actions:
            - {UTTER_GREET_ACTION}
            - {UTTER_BYE_ACTION}
            slots:
                slot_1:
                    type: bool
                slot_2:
                    type: bool
                slot_3:
                    type: bool
                slot_4:
                    type: bool
            """
        )
        events = [
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            SlotSet("slot_1", True),
            ActionExecuted(UTTER_GREET_ACTION),
            SlotSet("slot_2", True),
            SlotSet("slot_3", True),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(UTTER_GREET_ACTION),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            SlotSet("slot_4", True),
            ActionExecuted(UTTER_BYE_ACTION),
        ]
        training_story = TrackerWithCachedStates.from_events(
            "training story", evts=events, domain=domain, slots=domain.slots,
        )
        test_story = TrackerWithCachedStates.from_events(
            "training story", events[:-1], domain=domain, slots=domain.slots,
        )
        policy.train([training_story], domain)
        prediction = policy.predict_action_probabilities(
            test_story, domain, RegexInterpreter()
        )
        assert (
            domain.action_names_or_texts[
                prediction.probabilities.index(max(prediction.probabilities))
            ]
            == UTTER_BYE_ACTION
        )


class TestAugmentedMemoizationPolicy(TestMemoizationPolicy):
    """Test suite for AugmentedMemoizationPolicy."""

    @staticmethod
    def _policy_class_to_test() -> Type[PolicyGraphComponent]:
        return AugmentedMemoizationPolicy

    @pytest.mark.parametrize("max_history", [1, 2, 3, 4, None])
    def test_augmented_prediction(
        self,
        max_history: Optional[int],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ):
        policy = self.create_policy(
            featurizer=MaxHistoryTrackerFeaturizer(max_history=max_history),
            model_storage=model_storage,
            resource=resource,
            execution_context=execution_context,
        )

        GREET_INTENT_NAME = "greet"
        UTTER_GREET_ACTION = "utter_greet"
        UTTER_BYE_ACTION = "utter_goodbye"
        domain = Domain.from_yaml(
            f"""
                intents:
                - {GREET_INTENT_NAME}
                actions:
                - {UTTER_GREET_ACTION}
                - {UTTER_BYE_ACTION}
                slots:
                    slot_1:
                        type: bool
                        initial_value: true
                    slot_2:
                        type: bool
                    slot_3:
                        type: bool
                """
        )
        training_story = TrackerWithCachedStates.from_events(
            "training story",
            [
                ActionExecuted(UTTER_GREET_ACTION),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_3", True),
                ActionExecuted(UTTER_BYE_ACTION),
            ],
            domain=domain,
            slots=domain.slots,
        )
        test_story = TrackerWithCachedStates.from_events(
            "test story",
            [
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_1", False),
                ActionExecuted(UTTER_GREET_ACTION),
                ActionExecuted(UTTER_GREET_ACTION),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_2", True),
                ActionExecuted(UTTER_GREET_ACTION),
                UserUttered(intent={"name": GREET_INTENT_NAME}),
                ActionExecuted(UTTER_GREET_ACTION),
                SlotSet("slot_3", True),
                # ActionExecuted(UTTER_BYE_ACTION),
            ],
            domain=domain,
            slots=domain.slots,
        )
        policy.train([training_story], domain)
        prediction = policy.predict_action_probabilities(
            test_story, domain, RegexInterpreter()
        )
        assert (
            domain.action_names_or_texts[
                prediction.probabilities.index(max(prediction.probabilities))
            ]
            == UTTER_BYE_ACTION
        )


@pytest.mark.parametrize(
    "policy,supported_data",
    [
        (TEDPolicy, SupportedData.ML_DATA),
        (RulePolicy, SupportedData.ML_AND_RULE_DATA),
        (MemoizationPolicy, SupportedData.ML_DATA),
    ],
)
def test_supported_data(policy: Type[Policy], supported_data: SupportedData):
    assert policy.supported_data() == supported_data


class OnlyRulePolicy(Policy):
    """Test policy that supports only rule-based training data."""

    @staticmethod
    def supported_data() -> SupportedData:
        return SupportedData.RULE_DATA


@pytest.mark.parametrize(
    "policy,n_rule_trackers,n_ml_trackers",
    [
        (TEDPolicy(), 0, 3),
        (RulePolicy(), 2, 3),
        (OnlyRulePolicy, 2, 0),  # policy can be passed as a `type` as well
    ],
)
def test_get_training_trackers_for_policy(
    policy: Policy, n_rule_trackers: int, n_ml_trackers: int
):
    # create five trackers (two rule-based and three ML trackers)
    trackers = [
        DialogueStateTracker("id1", slots=[], is_rule_tracker=True),
        DialogueStateTracker("id2", slots=[], is_rule_tracker=False),
        DialogueStateTracker("id3", slots=[], is_rule_tracker=False),
        DialogueStateTracker("id4", slots=[], is_rule_tracker=True),
        DialogueStateTracker("id5", slots=[], is_rule_tracker=False),
    ]

    trackers = SupportedData.trackers_for_policy(policy, trackers)

    rule_trackers = [tracker for tracker in trackers if tracker.is_rule_tracker]
    ml_trackers = [tracker for tracker in trackers if not tracker.is_rule_tracker]

    assert len(rule_trackers) == n_rule_trackers
    assert len(ml_trackers) == n_ml_trackers
