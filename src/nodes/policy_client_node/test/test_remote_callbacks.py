"""Exercise the real ROS adapter callbacks with a controllable RPC executor.

ROS message types are real; no ROS context, sockets, robot, or remote host is started.
"""
import json
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
from std_msgs.msg import String
from ur_msgs.msg import ControlStatus

from policy_client_node import policy_client_node as module
from sharpa_interface.server.action import parse_policy_action
from test_policy_client_core import action, _metadata_format


class Pool:
    def __init__(self, **kwargs):
        self.jobs = []

    def submit(self, call, *args, **kwargs):
        future = Future()
        self.jobs.append((future, call, args, kwargs))
        return future

    def finish(self, result=None, error=None):
        future, call, args, kwargs = self.jobs.pop(0)
        if error:
            future.set_exception(error)
        else:
            future.set_result(result if result is not None else call(*args, **kwargs))

    def shutdown(self, **kwargs):
        pass


class Node:
    def __init__(self, name):
        self.parameters = {}
        self.messages = {}
        self.guard_callbacks = []

    def declare_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    def create_publisher(self, cls, topic, depth):
        messages = self.messages.setdefault(topic, [])
        return SimpleNamespace(publish=messages.append)

    def create_subscription(self, *args, **kwargs):
        return None

    def create_guard_condition(self, callback):
        return SimpleNamespace(trigger=lambda: self.guard_callbacks.append(callback))

    def spin_guards(self):
        while self.guard_callbacks:
            self.guard_callbacks.pop(0)()

    def create_timer(self, *args, **kwargs):
        return None

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1000))

    def get_logger(self):
        return SimpleNamespace(warning=lambda message: None)

    def destroy_node(self):
        pass


class Client:
    def __init__(self, config):
        self.closed = False

    def close(self):
        self.closed = True

    def connect(self):
        return {'metadata_format': _metadata_format('initial'), 'model_id': 'model', 'task_id': 'task', 'run_id': 'run'}

    def reset(self, session_id, request_id):
        return {'metadata_format': _metadata_format('reset-format')}

    def infer_action(self, payload, **kwargs):
        raise AssertionError('test must supply the response explicitly')


@pytest.fixture
def run_node(monkeypatch):
    import rclpy
    import rclpy.node
    import rclpy.executors
    import concurrent.futures
    import sharpa_interface.server.client
    import ur_common.camera_transport

    monkeypatch.setattr(rclpy, 'init', lambda **kwargs: None)
    monkeypatch.setattr(rclpy, 'ok', lambda: False)
    monkeypatch.setattr(rclpy.node, 'Node', Node)
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', Pool)
    monkeypatch.setattr(sharpa_interface.server.client, 'ServerClient', Client)
    monkeypatch.setattr(ur_common.camera_transport, 'SharedCameraReader',
                        lambda **kwargs: SimpleNamespace(close=lambda: None, read=lambda: None))

    def run(check):
        class Executor:
            def __init__(self, **kwargs):
                pass

            def add_node(self, node):
                self.node = node

            def spin(self):
                check(self.node)

            def shutdown(self):
                pass

        monkeypatch.setattr(rclpy.executors, 'MultiThreadedExecutor', Executor)
        module.main()
    return run


def ready(node):
    node._model_available = True
    node._started = True
    node._metadata = {'metadata_format': _metadata_format('initial')}
    node._control_ready = node._machine_control_ready = node._actor_gate_running = True
    node._core.update_actor_gate(True)
    node._observations.build = lambda *args, **kwargs: {'timestamp_ns': 1000}
    node._hil = {'rollout_recording': {'rollout_id': 'episode-a'}}


def response(node):
    pending = node._pending
    data = action(node._core.session_id, pending.request_id)
    parsed = parse_policy_action(data, expected_session_id=node._core.session_id,
                                 expected_request_id=pending.request_id)
    return replace(parsed, next_metadata_format=_metadata_format('next-format')), 0.05


def events(node):
    return [json.loads(message.data) for message in node.messages['/policy/inference']]


def control(paused=False, running=True):
    return ControlStatus(active_source=3 if running else 1, ur_state=2,
                         sharpa_state=1, execution_paused=paused)


def test_async_response_publishes_action_and_lossless_journal(run_node):
    def check(node):
        ready(node)
        node._tick()
        assert node._busy
        assert not node.messages['/policy/executable_action']
        node._inference_pool.finish(response(node))
        node._tick()
        assert len(node.messages['/policy/executable_action']) == 1
        requested, completed = events(node)
        assert requested['event'] == 'requested'
        assert completed['event'] == 'completed'
        assert requested['request_id'] == completed['request_id']
        assert requested['rollout_id'] == completed['rollout_id'] == 'episode-a'
        assert completed['result']['left_wrist']['eef']['shape'] == [2, 9]
        assert node._metadata['metadata_format']['format_id'] == 'next-format'
    run_node(check)


@pytest.mark.parametrize('change', ['pause', 'stop', 'provider', 'hil', 'model'])
def test_control_change_cancels_old_response_even_if_control_returns(run_node, change):
    def check(node):
        ready(node)
        node._tick()
        result = response(node)
        if change in ('pause', 'stop'):
            node._control_status(control(paused=change == 'pause', running=change != 'stop'))
            node._control_status(control())
        elif change == 'provider':
            node._provider_status(String(data='replay'))
            node._provider_status(String(data='policy'))
        elif change == 'hil':
            node._hil_status(String(data=json.dumps({'active': True})))
            node._hil_status(String(data=json.dumps({'active': False})))
            node._control_status(control())
        else:
            node._model_status(String(data=json.dumps({'generation': 2, 'phase': 'running'})))
        # Recorder can close immediately; the outstanding network RPC is not awaited.
        assert [item['event'] for item in events(node)] == ['requested', 'cancelled']
        node._hil['rollout_recording'] = {'rollout_id': 'episode-b'}
        node._inference_pool.finish(result)
        node._tick()
        assert not node.messages['/policy/executable_action']
        assert not node._busy
        assert len(events(node)) == 2
        assert events(node)[-1]['rollout_id'] == 'episode-a'
    run_node(check)


def test_pause_discards_advanced_server_format_and_resets_before_resuming(run_node):
    def check(node):
        ready(node)
        node._tick()
        result = response(node)
        node._control_status(control(paused=True))
        node._inference_pool.finish(result)
        node._tick()
        node._tick()
        assert not node._inference_pool.jobs  # no reset or infer while paused
        node._control_status(control())
        node._tick()
        assert node._service_context[0] == 'reset'
        node._inference_pool.finish()
        node._tick()
        assert node._metadata['metadata_format']['format_id'] == 'reset-format'
        assert node._control_ready
        assert node._core.session_id != 'ur-sharpa'
    run_node(check)


def test_connect_and_reset_do_not_block_control_callbacks(run_node):
    def check(node):
        node._model_available = True
        node._expected_model, node._expected_task_id, node._expected_run_id = 'model', 'task', 'run'
        node._tick()
        assert node._service_context[0] == 'connect'
        node._control_status(control())
        node._inference_pool.finish()
        node._tick()
        node._tick()
        assert node._service_context[0] == 'reset'
        node._control_status(control(running=False))
        node._inference_pool.finish()
        node._tick()
        assert not node._control_ready
        assert node._rollout_reset_pending
        assert not node.messages['/policy/executable_action']
    run_node(check)


def test_failed_inference_closes_journal_request_and_publishes_fault(run_node):
    def check(node):
        ready(node)
        node._tick()
        node._inference_pool.finish(error=ConnectionError('remote disconnected'))
        node._tick()
        assert [item['event'] for item in events(node)] == ['requested', 'failed']
        assert node._phase == 'fault'
        assert not node._busy
        assert node._core.ready
    run_node(check)


def test_paused_prefetch_keeps_active_chunk_until_done_before_session_reset(run_node):
    from ur_msgs.msg import ExecutionDone

    def check(node):
        ready(node)
        node._tick()
        parsed, latency = response(node)
        parsed = replace(parsed, execution=replace(parsed.execution, rtc=1))
        node._inference_pool.finish((parsed, latency))
        node._tick()
        node._done_msg(ExecutionDone(
            session_id=parsed.session_id, request_id=parsed.request_id,
            action_id=parsed.action_id, revision=parsed.revision,
            execute_start=0, execute_length=2, executed_steps=1,
            execute_fetch=True, execute_done=False, success=True))
        assert node._core.executing and node._core.inference_pending
        next_response = response(node)
        node._control_status(control(paused=True))
        node._inference_pool.finish(next_response)
        node._tick()
        node._control_status(control())
        node._tick()
        assert not node._inference_pool.jobs
        assert node._core.executing
        node._done_msg(ExecutionDone(
            session_id=parsed.session_id, request_id=parsed.request_id,
            action_id=parsed.action_id, revision=parsed.revision,
            execute_start=0, execute_length=2, executed_steps=2,
            execute_fetch=False, execute_done=True, success=True))
        node._tick()
        assert node._service_context[0] == 'reset'
        assert [event['event'] for event in events(node)] == ['requested', 'completed', 'requested', 'cancelled']
    run_node(check)


def test_rpc_failure_stays_faulted_until_explicit_stop_start(run_node):
    def check(node):
        ready(node)
        node._tick()
        node._inference_pool.finish(error=ConnectionError('offline'))
        node._tick()
        node._tick()
        assert not node._inference_pool.jobs
        assert node._phase == 'fault'
        node._control_status(control(running=False))
        node._control_status(control())
        node._tick()
        assert node._service_context[0] == 'reset'
    run_node(check)


def recording_status(node, rollout_id, *, ready=True, control_timestamp_ns=0):
    node._hil_status(String(data=json.dumps({
        'active': False, 'control_timestamp_ns': control_timestamp_ns,
        'rollout_recording': {'required': True, 'requested': True,
                              'recording': ready, 'rollout_id': rollout_id}})))


def test_first_inference_waits_for_recording_ack_before_creating_journal(run_node):
    from std_msgs.msg import Bool

    def check(node):
        ready(node)
        node._policy_recording_status(Bool(data=True))
        node._tick()
        assert not node._inference_pool.jobs
        assert not events(node)
        recording_status(node, 'episode-a', ready=False)
        node._tick()
        assert not events(node)
        recording_status(node, 'episode-a')
        node._tick()
        assert len(node._inference_pool.jobs) == 1
        assert events(node)[0]['rollout_id'] == 'episode-a'
    run_node(check)


def test_resume_requires_new_recording_id_and_current_control_timestamp(run_node):
    from std_msgs.msg import Bool

    def check(node):
        ready(node)
        node._policy_recording_status(Bool(data=True))
        recording_status(node, 'episode-a')
        node._tick()
        result = response(node)
        node._control_status(control(paused=True))
        resumed = control()
        resumed.timestamp_ns = 200
        node._control_status(resumed)
        node._inference_pool.finish(result)
        node._tick()  # discard cancelled response
        node._tick()  # reset server
        node._inference_pool.finish()
        node._tick()
        recording_status(node, 'episode-a', control_timestamp_ns=300)
        node._tick()
        assert not node._inference_pool.jobs
        recording_status(node, 'episode-b', control_timestamp_ns=150)
        node._tick()
        assert not node._inference_pool.jobs
        recording_status(node, 'episode-b', control_timestamp_ns=200)
        node._tick()
        assert len(node._inference_pool.jobs) == 1
        assert events(node)[-1]['event'] == 'requested'
        assert events(node)[-1]['rollout_id'] == 'episode-b'
    run_node(check)


def test_hil_requires_recording_without_checkbox_but_preview_does_not(run_node):
    def check(node):
        ready(node)
        node._execution_provider = 'hil'
        node._tick()
        assert not node._inference_pool.jobs
        node._execution_dry_run = True
        node._tick()
        assert len(node._inference_pool.jobs) == 1
    run_node(check)


def test_bridge_ack_arriving_before_local_control_status_is_accepted(run_node):
    def check(node):
        ready(node)
        node._machine_control_ready = False
        recording_status(node, 'new-episode', control_timestamp_ns=400)
        started = control()
        started.timestamp_ns = 400
        node._control_status(started)
        assert node._recording_ready()
    run_node(check)


def test_rpc_completion_wakes_executor_without_periodic_tick(run_node):
    def check(node):
        ready(node)
        node._tick()
        result, latency = response(node)
        result = replace(result, diagnostics={'client_response_wait_s': 0.04})
        node._inference_pool.finish((result, latency))
        # The worker completion must not publish in its callback thread.
        assert not node.messages['/policy/executable_action']
        assert node.guard_callbacks
        node.spin_guards()
        assert len(node.messages['/policy/executable_action']) == 1
        assert node._request_metrics.inference_count == 1
        completed = [json.loads(message.data) for message in
                     node.messages['/deploy/execution_event']
                     if json.loads(message.data)['event'] == 'model_inference_completed']
        assert completed[0]['diagnostics']['client_response_wait_s'] == 0.04
    run_node(check)


def test_completion_wakeup_preserves_cancelled_response_and_shutdown(run_node):
    def check(node):
        ready(node)
        node._tick()
        value = response(node)
        node._control_status(control(paused=True))
        node._inference_pool.finish(value)
        node.spin_guards()
        assert not node.messages['/policy/executable_action']
        assert node._request_metrics.inference_count == 0
        future = node._submit_rpc(lambda: None)
        node._close_completion_wakeups()
        node._inference_pool.finish()
        assert future.done() and not node.guard_callbacks
    run_node(check)
