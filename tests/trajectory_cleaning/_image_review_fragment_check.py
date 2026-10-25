"""Exercise a real dialog fragment rerun, which AppTest.run normally omits."""

from __future__ import annotations

from unittest.mock import patch

from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.scriptrunner import ScriptRunnerEvent
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.testing.v1.local_script_runner import LocalScriptRunner


def run_dialog_fragment(app):
    results = {"delta_types": []}

    class FragmentRunner(LocalScriptRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.queued = False

            def rerun_dialog(sender, event, **kw):
                if event == ScriptRunnerEvent.ENQUEUE_FORWARD_MSG and self.queued:
                    message = kw["forward_msg"]
                    if message.HasField("delta") and message.delta.HasField("new_element"):
                        results["delta_types"].append(message.delta.new_element.WhichOneof("type"))
                elif event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS and not self.queued:
                    self.queued = True
                    state = self.session_state._state
                    widget_id = state._key_id_mapper.get_id_from_key("image_review_fill")
                    metadata = state._new_widget_state.widget_metadata[widget_id]
                    states = WidgetStates(widgets=state.get_widget_states())
                    for widget in states.widgets:
                        if widget.id == widget_id:
                            widget.bool_value = not state["image_review_fill"]
                    results["fragment_id"] = metadata.fragment_id
                    self.request_rerun(RerunData(
                        widget_states=states, fragment_id_queue=[metadata.fragment_id]
                    ))
                elif event == ScriptRunnerEvent.FRAGMENT_STOPPED_WITH_SUCCESS:
                    results["fragment_success"] = True

            self.on_event.connect(rerun_dialog, weak=False)

    with patch("streamlit.testing.v1.app_test.LocalScriptRunner", FragmentRunner):
        app.run()
    return results


def dismiss_dialog(app):
    """Send the same trigger used by the popup X, backdrop, and Escape key."""
    result = {}

    class DismissRunner(LocalScriptRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.queued = False

            def dismiss(sender, event, **kw):
                if event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS and not self.queued:
                    state = self.session_state._state
                    self.queued = True
                    widget_id = next(
                        key for key, metadata in state._new_widget_state.widget_metadata.items()
                        if getattr(metadata.callback, "__name__", "") == "close_image_review"
                    )
                    states = WidgetStates(widgets=state.get_widget_states())
                    for widget in states.widgets:
                        if widget.id == widget_id:
                            widget.trigger_value = True
                    self.request_rerun(RerunData(widget_states=states))
                    result["dismiss_trigger_sent"] = True

            self.on_event.connect(dismiss, weak=False)

    with patch("streamlit.testing.v1.app_test.LocalScriptRunner", DismissRunner):
        app.run()
    return result
