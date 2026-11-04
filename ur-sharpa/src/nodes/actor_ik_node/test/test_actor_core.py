

def test_actor_does_not_execute_under_teleop_ownership():
    from types import SimpleNamespace
    from actor_ik_node.actor_ik_node import actor_execution_requested
    status = SimpleNamespace(active_source=2, ur_state=2, sharpa_state=1,
                             execution_scope=0, execution_dry_run=False)
    assert not actor_execution_requested(status)
    status.active_source=3
    assert actor_execution_requested(status)
    status.ur_state=0
    assert not actor_execution_requested(status)
    status.execution_scope=1
    assert actor_execution_requested(status)
    status.active_source=2
    assert not actor_execution_requested(status)
    status.execution_dry_run=True
    assert not actor_execution_requested(status)
