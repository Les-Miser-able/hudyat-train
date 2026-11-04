import time
import queue
import threading
from types import SimpleNamespace

import pytest
from hudyat_train.experiments import Registry
from hudyat_train.storage import write_json


def test_live_preview_refresh_is_fast_enough_for_30_fps():
    from hudyat_train.gui import App

    scheduled = []
    app = App.__new__(App)
    app.events = queue.Queue()
    app.live_frame = None
    app.live_frame_lock = threading.Lock()
    app.live_ready = True
    app.window = SimpleNamespace(after=lambda delay, callback: scheduled.append(delay))
    app._poll()
    assert scheduled[-1] <= 33, f"Preview polling takes {scheduled[-1]} ms"


def test_desktop_runs_scan_in_background_and_updates_coverage(project):
    from hudyat_train.gui import App, create_window
    try:
        window = create_window()
    except Exception as error:
        pytest.skip(f"Desktop display unavailable: {error}")
    window.withdraw()
    try:
        app = App(window, project.root)
        assert len(app.tabs.tabs()) == 7
        app.scan()
        assert app.process is not None
        deadline = time.monotonic() + 15
        while app.process is not None and time.monotonic() < deadline:
            window.update()
            time.sleep(0.02)
        assert app.process is None
        assert app.status.get() == "Completed."
        assert len(app.coverage.get_children()) == 9
        app.coverage.selection_set("alphabet")
        window.update()
        assert len(app.sign_counts.get_children()) == 26
    finally:
        window.destroy()


def test_results_tables_and_live_target_controls_show_saved_data(project):
    from hudyat_train.gui import App, create_window
    try:
        window = create_window()
    except Exception as error:
        pytest.skip(f"Desktop display unavailable: {error}")
    window.withdraw()
    try:
        run = project.path("runs_dir") / "model"
        write_json(run / "classes.json", ["A", "B"])
        record = Registry(project.root).save(run, "training", {"status": "complete", "category": "alphabet", "variant": "sequence"},
                                             {"accuracy": 0.75, "macro_f1": 0.7, "per_class": [{"label": "A", "support": 2, "precision": 1, "recall": 0.5, "f1": 2 / 3}]})
        app = App(window, project.root)
        app.results_view.refresh()
        app.results_view.overview.selection_set(record["id"])
        window.update()
        assert len(app.results_view.per_class.get_children()) == 1
        values = app.results_view.overview.item(record["id"], "values")
        assert "75.0" in values and "70.0" in values
        app.choose_test_model(run)
        assert list(app.target_box["values"]) == ["(unscored)", "A", "B"]
        assert app.live_mode.get() == "automatic"
        assert str(app.capture_button["state"]) == "disabled"
        assert app.tabs.select() == str(app.testing_tab)
    finally:
        window.destroy()


def test_desktop_augmentation_controls_reach_the_training_job(project, monkeypatch):
    from hudyat_train.gui import App, create_window
    try:
        window = create_window()
    except Exception as error:
        pytest.skip(f"Desktop display unavailable: {error}")
    window.withdraw()
    try:
        app = App(window, project.root)
        commands = []
        monkeypatch.setattr(app, "_start", lambda arguments, kind: commands.append(arguments))
        assert app.augmentation_enabled.get()
        assert app.augmentation_copies.get() == "5"
        assert "6 training samples" in app.augmentation_summary.get()
        assert app.mirroring.get() == "All signs"
        assert "3 mirrored copies per original" in app.augmentation_summary.get()
        app.train()
        assert commands[-1][commands[-1].index("--mirroring") + 1] == "all"
        app.development.set(True)
        app.augmentation_copies.set("3")
        app.train()
        assert commands[-1][-2:] == ["3", "--development"]
        assert "--augmentation-copies" in commands[-1]
        app.mirroring.set("Approved signs only")
        assert "2 mirrored copies per approved original" in app.augmentation_summary.get()
        app.train()
        assert commands[-1][commands[-1].index("--mirroring") + 1] == "approved"
        app.mirroring.set("Off")
        assert "mirroring is off" in app.augmentation_summary.get()
        app.train()
        assert commands[-1][commands[-1].index("--mirroring") + 1] == "off"
        app.augmentation_enabled.set(False)
        app._augmentation_controls()
        assert str(app.augmentation_box["state"]) == "disabled"
        assert str(app.mirroring_box["state"]) == "disabled"
        app.train()
        assert commands[-1][-2:] == ["0", "--development"]
    finally:
        window.destroy()


def test_desktop_hudyat_simulator_passes_selected_models_and_settings(project, monkeypatch):
    from hudyat_train.gui import App, create_window
    try:
        window = create_window()
    except Exception as error:
        pytest.skip(f"Desktop display unavailable: {error}")
    window.withdraw()
    try:
        app = App(window, project.root)
        commands = []
        monkeypatch.setattr(app, "_start", lambda arguments, kind: commands.append((arguments, kind)))
        assert app.test_modes.tab(2, "text") == "Simulate HUDYAT"
        assert app.simulation_profile.get() == "baseline"
        app.test_run.set(str(project.root / "selected-model"))
        app.simulation_candidate.set(str(project.root / "candidate-model"))
        app.simulation_settings.set(str(project.root / "lab-results.json"))
        app.simulation_profile.set("enhanced")
        app.live_target.set("A")
        app.start_simulation()
        arguments, kind = commands[-1]
        assert kind == "simulation" and arguments[0] == "simulate"
        assert arguments[arguments.index("--run") + 1] == str(project.root / "selected-model")
        assert arguments[arguments.index("--candidate") + 1] == str(project.root / "candidate-model")
        assert arguments[arguments.index("--settings-file") + 1] == str(project.root / "lab-results.json")
        assert arguments[arguments.index("--profile") + 1] == "enhanced"
        assert arguments[arguments.index("--hudyat") + 1] == app.hudyat_path.get()
        assert arguments[arguments.index("--target") + 1] == "A"
    finally:
        window.destroy()
