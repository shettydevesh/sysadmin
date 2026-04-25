"""Tests for typed OpenEnv-facing models and splits."""

from sysadmin_env.models import Action, Observation, State
from sysadmin_env.scenarios import TRAIN_SCENARIO_IDS, VAL_SCENARIO_IDS, list_scenarios


def test_action_observation_state_models():
    action = Action(command="df -h")
    observation = Observation(output="ok", reward=0.1, metadata={"episode_id": "ep"})
    state = State(episode_id="ep", scenario_id="disk_full")

    assert action.command == "df -h"
    assert observation.metadata["episode_id"] == "ep"
    assert state.step_count == 0


def test_train_and_eval_splits_are_disjoint_and_registered():
    all_scenarios = set(list_scenarios())

    assert set(TRAIN_SCENARIO_IDS).isdisjoint(VAL_SCENARIO_IDS)
    assert set(TRAIN_SCENARIO_IDS).issubset(all_scenarios)
    assert set(VAL_SCENARIO_IDS).issubset(all_scenarios)
