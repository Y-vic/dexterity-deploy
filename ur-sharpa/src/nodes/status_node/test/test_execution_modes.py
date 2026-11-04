from status_node.gates import GateMachine, UrState, SharpaState, ExecutionScope


def test_policy_defaults_to_no_recording_and_hil_is_a_distinct_mode():
    machine = GateMachine(mode='deploy')
    assert machine.execution_provider == 'policy'
    assert machine.policy_recording is False
    assert machine.attempt_set_policy_recording(True)
    assert machine.status_payload()['policy_recording'] is True
    machine.execution_scope = ExecutionScope.HANDS_ONLY
    assert machine.attempt_set_execution_provider('hil')
    assert machine.execution_scope == ExecutionScope.FULL
    assert not machine.attempt_set_policy_recording(False)
    assert machine.attempt_set_execution_provider('policy')
    assert machine.attempt_set_policy_recording(False)
    assert machine.policy_recording is False


def test_running_preview_and_starting_block_mode_and_recording_changes():
    for fields in (dict(ur_state=UrState.RUNNING, sharpa_state=SharpaState.RUNNING),
                   dict(execution_dry_run=True), dict(execution_start_pending=True)):
        machine = GateMachine(mode='deploy', **fields)
        assert not machine.attempt_set_execution_provider('hil')
        assert not machine.attempt_set_policy_recording(True)
        assert machine.execution_provider == 'policy' and not machine.policy_recording


def test_teleop_does_not_accept_deployment_recording_settings():
    machine = GateMachine(mode='teleop')
    assert not machine.attempt_set_policy_recording(True)
    assert not machine.attempt_set_execution_provider('hil')
