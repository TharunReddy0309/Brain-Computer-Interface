import os
import sys
import csv
import json
import random
import datetime

from psychopy import visual, core, event, gui, sound, logging

try:
    from pylsl import StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False


HEADSET_CHANNELS = 14
SAMPLING_RATE_HZ = 128
MU_BAND_HZ = (8, 12)
BETA_BAND_HZ = (18, 26)
RECOMMENDED_PROCESSING_BAND_HZ = (4, 40)
POWERLINE_NOTCH_HZ = 50

REFERENCE_DATASET_CHANNELS = 22
REFERENCE_DATASET_EOG_CHANNELS = 3
REFERENCE_DATASET_SAMPLING_RATE_HZ = 250
REFERENCE_DATASET_REFERENCE = "Left mastoid (reference) / Right mastoid (ground)"

T_FIXATION_ONSET = 0.0
T_WARNING_TONE = 0.0
T_CUE_ONSET = 2.0
T_CUE_DURATION = 1.25
T_TRIAL_END = 6.0
CUE_OFFSET_TIME = T_CUE_ONSET + T_CUE_DURATION

ITI_DURATION = 2.0

CLASS_LABELS_DEFAULT = ["LEFT_HAND", "RIGHT_HAND", "FEET", "REST"]
CLASS_LABELS_TONGUE = ["LEFT_HAND", "RIGHT_HAND", "FEET", "TONGUE"]

CUE_DISPLAY_TEXT = {
    "LEFT_HAND": "\u2190",
    "RIGHT_HAND": "\u2192",
    "FEET": "\u2193\u2193",
    "REST": "REST",
    "TONGUE": "TONGUE",
}

CUE_EVENT_NAME = {
    "LEFT_HAND": "CUE_LEFT",
    "RIGHT_HAND": "CUE_RIGHT",
    "FEET": "CUE_FEET",
    "REST": "CUE_REST",
    "TONGUE": "CUE_TONGUE",
}

MI_RATING_SCALE = {
    "1": "Could not perform imagery",
    "2": "Very weak imagery",
    "3": "Moderate imagery",
    "4": "Strong imagery",
    "5": "Very clear/strong imagery",
}

OPERATOR_FLAG_KEYS = {
    "a": "MI_SUCCESS",
    "u": "MI_UNCERTAIN",
    "m": "PHYSICAL_MOVEMENT",
    "e": "EYE_ARTIFACT",
    "d": "DISTRACTION",
    "r": "TRIAL_REJECTED",
}

SOFTWARE_VERSION = "MI-EEG-PsychoPy v1.0"


def get_session_info():
    info = {
        "Subject ID": "S001",
        "Session ID": "S01",
        "Operator ID": "OP1",
        "Headset model": "Emotiv EPOC X",
        "Number of runs": 4,
        "Trials per class per run": 8,
        "Use TONGUE instead of REST (2a replication)": False,
        "Run baseline/EOG block": True,
        "Run familiarization/practice block": True,
        "Random seed (blank = auto)": "",
        "Fullscreen": True,
    }
    dlg = gui.DlgFromDict(
        dictionary=info,
        title="Motor Imagery EEG Acquisition - Session Setup",
        order=[
            "Subject ID", "Session ID", "Operator ID", "Headset model",
            "Number of runs", "Trials per class per run",
            "Use TONGUE instead of REST (2a replication)",
            "Run baseline/EOG block",
            "Run familiarization/practice block",
            "Random seed (blank = auto)", "Fullscreen",
        ],
    )
    if not dlg.OK:
        core.quit()
    return info


def get_acquisition_checklist():
    checklist = {
        "Headset connected to acquisition computer": False,
        "All intended EEG channels detected": False,
        "Sampling rate confirmed correct (128 Hz)": False,
        "Recording software receiving continuous EEG": False,
        "Electrode/contact quality checked (all channels)": False,
        "Event markers confirmed working (test marker sent)": False,
        "Live-signal inspection performed, signal stable": False,
        "Headset centered / all electrodes correctly positioned": False,
        "No channel shows persistent poor contact": False,
    }
    dlg = gui.DlgFromDict(
        dictionary=checklist,
        title="Pre-Experiment Acquisition & Electrode Checklist "
              "(Sec 1.3 / 2.2)",
    )
    if not dlg.OK:
        core.quit()
    if not all(checklist.values()):
        print("Checklist incomplete. Per protocol Sec 1.3/2.2, the "
              "experiment cannot begin until every item is confirmed.")
        core.quit()
    return checklist


def setup_directories(subject_id, session_id, base_dir="MI_EEG_DATASET"):
    root = os.path.join(base_dir, "Subject_%s" % subject_id,
                        "Session_%s" % session_id)
    paths = {
        "root": root,
        "Raw": os.path.join(root, "Raw"),
        "Events": os.path.join(root, "Events"),
        "Metadata": os.path.join(root, "Metadata"),
        "Quality": os.path.join(root, "Quality"),
        "Epochs": os.path.join(root, "Epochs"),
        "Dataset_Metadata": os.path.join(base_dir, "Dataset_Metadata"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


class EventLogger:

    EVENT_FIELDS = [
        "subject_id", "session_id", "run_id", "trial_id",
        "event_type", "class", "timestamp", "sample_index",
        "duration", "quality_flag",
    ]

    def __init__(self, subject_id, session_id, events_dir, fs=SAMPLING_RATE_HZ):
        self.subject_id = subject_id
        self.session_id = session_id
        self.fs = fs
        self.clock = core.Clock()

        fname = "%s_%s_Events.csv" % (subject_id, session_id)
        self.filepath = os.path.join(events_dir, fname)
        self._fh = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.EVENT_FIELDS)
        self._writer.writeheader()

        self.outlet = None
        if LSL_AVAILABLE:
            info = StreamInfo(
                name="PsychoPy_MI_Markers",
                type="Markers",
                channel_count=1,
                nominal_srate=0,
                channel_format="string",
                source_id="mi_eeg_protocol_%s_%s" % (subject_id, session_id),
            )
            self.outlet = StreamOutlet(info)

    def log(self, event_type, run_id="", trial_id="", class_label="",
            duration=0.0, quality_flag=""):
        timestamp = round(self.clock.getTime(), 3)
        sample_index = int(round(timestamp * self.fs))
        row = {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "run_id": run_id,
            "trial_id": trial_id,
            "event_type": event_type,
            "class": class_label,
            "timestamp": timestamp,
            "sample_index": sample_index,
            "duration": duration,
            "quality_flag": quality_flag,
        }
        self._writer.writerow(row)
        self._fh.flush()

        if self.outlet is not None:
            marker_str = "%s|%s|%s|%s" % (
                event_type, trial_id, class_label, timestamp)
            self.outlet.push_sample([marker_str])

        logging.exp("EVENT %s" % row)
        return timestamp, sample_index

    def close(self):
        self._fh.close()


class QualityLogger:
    FIELDS = [
        "subject_id", "session_id", "run_id", "trial_id", "class",
        "mi_rating", "operator_flag", "physical_movement",
        "eye_artifact", "distraction", "accepted",
    ]

    def __init__(self, subject_id, session_id, quality_dir):
        fname = "%s_%s_Quality.csv" % (subject_id, session_id)
        self.filepath = os.path.join(quality_dir, fname)
        self._fh = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def log(self, subject_id, session_id, run_id, trial_id, class_label,
            mi_rating, operator_flag, physical_movement, eye_artifact,
            distraction, accepted):
        row = {
            "subject_id": subject_id, "session_id": session_id,
            "run_id": run_id, "trial_id": trial_id, "class": class_label,
            "mi_rating": mi_rating, "operator_flag": operator_flag,
            "physical_movement": physical_movement,
            "eye_artifact": eye_artifact, "distraction": distraction,
            "accepted": accepted,
        }
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        self._fh.close()


def generate_run_order(classes, trials_per_class, rng):
    pool = classes * trials_per_class
    for _attempt in range(1000):
        rng.shuffle(pool)
        if all(pool[i] != pool[i + 1] for i in range(len(pool) - 1)):
            return list(pool)
    return list(pool)


def build_stimuli(win):
    fixation = visual.TextStim(win, text="+", height=0.08, color="white")
    cue = visual.TextStim(win, text="", height=0.15, color="white")
    instructions = visual.TextStim(
        win, text="", height=0.045, color="white", wrapWidth=1.4)
    try:
        warning_tone = sound.Sound(value=880, secs=0.2)
    except Exception:
        warning_tone = None
    return fixation, cue, instructions, warning_tone


def wait_for_key(keys=("space",), allow_escape=True):
    valid = list(keys) + (["escape"] if allow_escape else [])
    pressed = event.waitKeys(keyList=valid)
    if allow_escape and "escape" in pressed:
        core.quit()
    return pressed[0]


def check_escape():
    if "escape" in event.getKeys(["escape"]):
        core.quit()


def run_baseline_block(win, instructions, fixation, logger):
    segments = [
        ("BASELINE_EYES_OPEN", 120,
         "Baseline recording: EYES OPEN.\n\n"
         "Please look at the fixation cross and remain still.\n"
         "This will last 2 minutes.\n\nPress SPACE to begin."),
        ("BASELINE_EYES_CLOSED", 60,
         "Baseline recording: EYES CLOSED.\n\n"
         "Please close your eyes and remain still.\n"
         "This will last 1 minute. An audio tone will play at the end.\n\n"
         "Press SPACE to begin."),
        ("BASELINE_EYE_MOVEMENT", 60,
         "Baseline recording: EYE MOVEMENTS.\n\n"
         "Please move your eyes naturally (blinks, glances) for 1 minute.\n"
         "This helps identify eye-related artifacts.\n\n"
         "Press SPACE to begin."),
    ]
    for name, dur, text in segments:
        instructions.text = text
        instructions.draw()
        win.flip()
        wait_for_key(["space"])

        logger.log(name + "_START")
        clock = core.Clock()
        while clock.getTime() < dur:
            fixation.draw()
            win.flip()
            check_escape()
        logger.log(name + "_END")


def run_practice_block(win, classes, instructions, cue, fixation, warning_tone):
    instructions.text = (
        "FAMILIARIZATION\n\n"
        "You will now practice each motor-imagery class.\n"
        "When you see a cue, IMAGINE the movement -- do NOT physically "
        "move.\n\nClasses:\n" +
        "\n".join(["  %s -> %s" % (c, CUE_DISPLAY_TEXT[c]) for c in classes]) +
        "\n\nPress SPACE to begin practice."
    )
    instructions.draw()
    win.flip()
    wait_for_key(["space"])

    for practice_class in classes:
        instructions.text = (
            "Practice class: %s\n\nPress SPACE when ready." % practice_class)
        instructions.draw()
        win.flip()
        wait_for_key(["space"])

        fixation.draw()
        win.flip()
        if warning_tone is not None:
            try:
                warning_tone.play()
            except Exception:
                pass
        core.wait(T_CUE_ONSET)

        cue.text = CUE_DISPLAY_TEXT[practice_class]
        cue.draw()
        win.flip()
        core.wait(T_CUE_DURATION)

        fixation.draw()
        win.flip()
        core.wait(T_TRIAL_END - CUE_OFFSET_TIME)
        check_escape()

    instructions.text = (
        "Familiarization complete.\n\n"
        "Formal recording will now begin.\nPress SPACE to continue.")
    instructions.draw()
    win.flip()
    wait_for_key(["space"])


def run_trial(win, fixation, cue, class_label, run_id, trial_id, logger,
              warning_tone):
    logger.log("TRIAL_START", run_id, trial_id, class_label)

    trial_clock = core.Clock()

    fixation.draw()
    win.flip()
    if warning_tone is not None:
        try:
            warning_tone.play()
        except Exception:
            pass
    while trial_clock.getTime() < T_CUE_ONSET:
        check_escape()
        core.wait(0.005)

    cue.text = CUE_DISPLAY_TEXT[class_label]
    cue.draw()
    win.flip()
    logger.log(CUE_EVENT_NAME[class_label], run_id, trial_id, class_label,
               duration=T_CUE_DURATION)
    logger.log("MI_START", run_id, trial_id, class_label)

    while trial_clock.getTime() < CUE_OFFSET_TIME:
        check_escape()
        core.wait(0.005)

    fixation.draw()
    win.flip()
    while trial_clock.getTime() < T_TRIAL_END:
        check_escape()
        core.wait(0.005)

    logger.log("MI_END", run_id, trial_id, class_label)
    logger.log("TRIAL_END", run_id, trial_id, class_label)

    win.flip()
    core.wait(ITI_DURATION)


def collect_mi_rating(win, instructions):
    lines = ["How clear was your motor imagery on that trial?\n"]
    for k, v in sorted(MI_RATING_SCALE.items()):
        lines.append("  %s = %s" % (k, v))
    instructions.text = "\n".join(lines)
    instructions.draw()
    win.flip()
    key = wait_for_key(list(MI_RATING_SCALE.keys()))
    return key


def collect_operator_flag(win, instructions):
    lines = ["OPERATOR: trial-quality flag\n"]
    for k, v in OPERATOR_FLAG_KEYS.items():
        lines.append("  %s = %s" % (k, v))
    instructions.text = "\n".join(lines)
    instructions.draw()
    win.flip()
    key = wait_for_key(list(OPERATOR_FLAG_KEYS.keys()))
    return OPERATOR_FLAG_KEYS[key]


def write_metadata(paths, info, classes, trial_order_per_run, seed,
                    checklist, session_start_iso, acceptance_summary):
    metadata = {
        "subject_id": info["Subject ID"],
        "session_id": info["Session ID"],
        "date": session_start_iso,
        "operator_id": info["Operator ID"],
        "headset_model": info["Headset model"],
        "number_of_channels": HEADSET_CHANNELS,
        "sampling_frequency_hz": SAMPLING_RATE_HZ,
        "electrode_montage": "Manufacturer-defined montage of selected "
                              "headset (Sec 2.1) - record exact channel "
                              "names used for this headset here.",
        "reference_configuration": "As implemented by the selected "
                                    "headset (Sec 2.3) - NOT assumed to be "
                                    "the BCI IV 2a mastoid reference.",
        "ground_configuration": "As implemented by the selected headset.",
        "trial_count_per_run": len(trial_order_per_run[0]) if trial_order_per_run else 0,
        "number_of_runs": len(trial_order_per_run),
        "total_trials_session": sum(len(r) for r in trial_order_per_run),
        "class_definitions": classes,
        "trial_randomization_seed": seed,
        "trial_order_per_run": trial_order_per_run,
        "software_version": SOFTWARE_VERSION,
        "recording_notes": "",
        "rejected_channels": [],
        "rejected_trials": [],
        "acquisition_checklist": checklist,
        "session_acceptance_summary": acceptance_summary,
        "reference_dataset_note": {
            "name": "BCI Competition IV 2a",
            "channels": REFERENCE_DATASET_CHANNELS,
            "eog_channels": REFERENCE_DATASET_EOG_CHANNELS,
            "sampling_rate_hz": REFERENCE_DATASET_SAMPLING_RATE_HZ,
            "reference": REFERENCE_DATASET_REFERENCE,
            "note": "Documented for comparison only; NOT the configuration "
                    "used for this project's real-headset acquisition.",
        },
    }
    fname = "%s_%s_Metadata.json" % (info["Subject ID"], info["Session ID"])
    fpath = os.path.join(paths["Metadata"], fname)
    with open(fpath, "w") as fh:
        json.dump(metadata, fh, indent=2)
    return fpath


def compute_acceptance_summary(checklist, n_trials_completed,
                                n_trials_target, n_rejected):
    acquisition_valid = all(checklist.values())
    sufficient_valid_trials = (n_trials_completed - n_rejected) >= 0.8 * n_trials_target
    if acquisition_valid and sufficient_valid_trials:
        status = "ACCEPTED"
    elif acquisition_valid and n_trials_completed > 0:
        status = "PARTIALLY ACCEPTED"
    else:
        status = "REJECTED"
    return {
        "status": status,
        "acquisition_valid": acquisition_valid,
        "trials_completed": n_trials_completed,
        "trials_target": n_trials_target,
        "trials_rejected": n_rejected,
        "sufficient_valid_trials": sufficient_valid_trials,
        "note": "Automatic pre-check only. Final ACCEPTED / PARTIALLY "
                "ACCEPTED / REJECTED status must still be confirmed against "
                "Sec 7.1-7.4 channel- and trial-level inspection of the raw "
                "EEG recorded by the headset software.",
    }


def main():
    info = get_session_info()
    checklist = get_acquisition_checklist()

    seed_raw = info["Random seed (blank = auto)"]
    seed = int(seed_raw) if str(seed_raw).strip() != "" else random.randint(0, 999999)
    rng = random.Random(seed)

    classes = (CLASS_LABELS_TONGUE
               if info["Use TONGUE instead of REST (2a replication)"]
               else CLASS_LABELS_DEFAULT)

    n_runs = int(info["Number of runs"])
    trials_per_class = int(info["Trials per class per run"])

    paths = setup_directories(info["Subject ID"], info["Session ID"])

    win = visual.Window(
        fullscr=bool(info["Fullscreen"]), color="black", units="height",
        allowGUI=False,
    )
    fixation, cue, instructions, warning_tone = build_stimuli(win)

    logger = EventLogger(info["Subject ID"], info["Session ID"], paths["Events"])
    quality = QualityLogger(info["Subject ID"], info["Session ID"], paths["Quality"])

    session_start_iso = datetime.datetime.now().isoformat()
    logger.log("SESSION_START")

    if not LSL_AVAILABLE:
        instructions.text = (
            "NOTE: pylsl is not installed.\n"
            "Event markers will be logged to the Events CSV file only and "
            "will NOT be streamed live for LSL-based EEG synchronization.\n"
            "Install pylsl for full Sec 4.5 synchronization support.\n\n"
            "Press SPACE to continue.")
        instructions.draw()
        win.flip()
        wait_for_key(["space"])

    if info["Run baseline/EOG block"]:
        run_baseline_block(win, instructions, fixation, logger)

    if info["Run familiarization/practice block"]:
        run_practice_block(win, classes, instructions, cue, fixation, warning_tone)

    trial_order_per_run = [
        generate_run_order(classes, trials_per_class, rng)
        for _ in range(n_runs)
    ]

    n_trials_completed = 0
    n_trials_rejected = 0
    n_trials_target = n_runs * len(classes) * trials_per_class

    for run_idx in range(1, n_runs + 1):
        run_id = "R%02d" % run_idx
        logger.log("RUN_START", run_id)

        instructions.text = (
            "Run %d of %d\n\nRemain still, avoid talking, follow the "
            "cues only.\n\nPress SPACE to begin this run." % (run_idx, n_runs))
        instructions.draw()
        win.flip()
        wait_for_key(["space"])

        order = trial_order_per_run[run_idx - 1]
        for t_idx, class_label in enumerate(order, start=1):
            trial_id = "T%03d" % (
                (run_idx - 1) * len(order) + t_idx)

            run_trial(win, fixation, cue, class_label, run_id, trial_id,
                      logger, warning_tone)

            rating = collect_mi_rating(win, instructions)
            op_flag = collect_operator_flag(win, instructions)
            accepted = op_flag not in ("TRIAL_REJECTED",)
            if not accepted:
                logger.log("TRIAL_REJECTED", run_id, trial_id, class_label,
                           quality_flag=op_flag)
                n_trials_rejected += 1

            quality.log(
                info["Subject ID"], info["Session ID"], run_id, trial_id,
                class_label, rating, op_flag,
                physical_movement=(op_flag == "PHYSICAL_MOVEMENT"),
                eye_artifact=(op_flag == "EYE_ARTIFACT"),
                distraction=(op_flag == "DISTRACTION"),
                accepted=accepted,
            )
            n_trials_completed += 1

        logger.log("RUN_END", run_id)

        if run_idx < n_runs:
            instructions.text = (
                "End of run %d.\n\nTake a short break.\n"
                "Press SPACE when ready for the next run." % run_idx)
            instructions.draw()
            win.flip()
            wait_for_key(["space"])

    logger.log("SESSION_END")

    acceptance_summary = compute_acceptance_summary(
        checklist, n_trials_completed, n_trials_target, n_trials_rejected)

    meta_path = write_metadata(
        paths, info, classes, trial_order_per_run, seed, checklist,
        session_start_iso, acceptance_summary)

    logger.close()
    quality.close()

    instructions.text = (
        "Session complete.\n\n"
        "Status: %s\n"
        "Trials completed: %d / %d (rejected: %d)\n\n"
        "Event file:    %s\n"
        "Quality file:  %s\n"
        "Metadata file: %s\n\n"
        "Press SPACE to exit." % (
            acceptance_summary["status"], n_trials_completed,
            n_trials_target, n_trials_rejected,
            logger.filepath, quality.filepath, meta_path)
    )
    instructions.draw()
    win.flip()
    wait_for_key(["space"])

    win.close()
    core.quit()


if __name__ == "__main__":
    main()
