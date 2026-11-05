"""Auto-save live rollouts using the existing demonstration recorder."""
import time


class RolloutRecorder:
    def __init__(self, core, save, saving):
        self.core, self.save, self.saving = core, save, saving
        self.requested = self.rollout_id = self.finished_id = ''
        self.seen = 0.0
        self.stop_at = None
        self.error = ''
        self.category = ''
        self.classified_id = ''
        self.trace_paths = {}
        self.audit_required = {}

    def request(self, value, now=None):
        if type(value.get('active')) is not bool:
            raise ValueError('rollout.active must be boolean')
        identity = value.get('rollout_id', '')
        if not isinstance(identity, str) or (value['active'] and not identity):
            raise ValueError('active rollout requires an ID')
        if identity:
            self.audit_required[identity] = bool(value.get('execution_audit'))
        if identity and value.get('actor_trace_path'):
            self.trace_paths[identity] = value['actor_trace_path']
        if value.get('finished_rollout_id') == self.rollout_id and self.rollout_id and value.get('termination'):
            self.core.termination = dict(value['termination'])
        category = value.get('category', '')
        if category not in ('', 'fail'):
            raise ValueError('unsupported rollout category')
        if category and value.get('finished_rollout_id') == self.rollout_id and self.rollout_id:
            self.category = category
        self.requested = identity if value['active'] else ''
        self.seen = time.monotonic() if now is None else now

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        wanted = self.requested if now - self.seen < 2.0 else ''
        if self.saving():
            return
        if (not self.core.recording and self.category and self.rollout_id
                and self.finished_id == self.rollout_id and self.classified_id != self.rollout_id):
            self.classified_id = self.rollout_id
            try:
                self.core.classify(self.category)
            except Exception as exc:
                self.error = 'rollout classification failed: ' + str(exc)
        if self.core.recording:
            if wanted == self.rollout_id:
                self.stop_at = None
                return
            if self.stop_at is None:
                self.stop_at = now + 0.25  # Keep final hardware feedback/camera frames.
                if self.requested == self.rollout_id:
                    self.core._note_quality_error('rollout heartbeat expired')
                    self.core.termination = dict(outcome='truncated', timestamp_ns=time.time_ns(),
                        action_id=None, annotation_source='recorder_heartbeat_timeout', annotated_unix_ns=time.time_ns())
            if now >= self.stop_at:
                self.finished_id = self.rollout_id
                self.save()
            return
        if wanted and wanted != self.finished_id:
            self.rollout_id = wanted
            self.error = ''
            self.category = ''
            self.stop_at = None
            try:
                if not self.core.start():
                    raise RuntimeError('recorder is still finalizing a previous episode')
                self.core.inference_rollout_id = wanted
                self.core.actor_trace_path = self.trace_paths.get(wanted, '')
                self.core.execution_audit_required = self.audit_required.get(wanted, False)
                self.audit_required = {wanted: self.core.execution_audit_required}
                self.trace_paths = {wanted: self.core.actor_trace_path}
            except Exception as exc:
                self.error = str(exc)
                self.finished_id = wanted

    def status(self):
        return dict(rollout_id=self.rollout_id, rollout_error=self.error,
                    category=self.category if self.classified_id == self.rollout_id and not self.error else '')
