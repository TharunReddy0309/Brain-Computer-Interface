import os
import csv
import json
import datetime
from psychopy import visual, core, event, gui, logging, monitors

my_monitor = monitors.Monitor("default_mon")
my_monitor.setWidth(30)      
my_monitor.setDistance(60)   

try:
    from pylsl import StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False

SAMPLING_RATE_HZ = 128          
DUR_RELAX_LONG = 25.0
DUR_RELAX_SHORT = 5.0
DUR_TASK = 25.0
DUR_INSTRUCTIONS = 10.0
NUM_TRIALS = 3

TASKS = ["Stroop", "Mirror_Image", "Arithmetic"]
SOFTWARE_VERSION = "SAM40-PsychoPy v1.6"

STROOP_TRIALS = [
    [
        {"word": "RED", "color": "blue"},
        {"word": "GREEN", "color": "yellow"},
        {"word": "BLUE", "color": "red"},
        {"word": "YELLOW", "color": "green"},
        {"word": "PURPLE", "color": "orange"}
    ],
    [
        {"word": "ORANGE", "color": "green"},
        {"word": "RED", "color": "purple"},
        {"word": "GREEN", "color": "blue"},
        {"word": "BLUE", "color": "yellow"},
        {"word": "YELLOW", "color": "red"}
    ],
    [
        {"word": "BLUE", "color": "orange"},
        {"word": "YELLOW", "color": "red"},
        {"word": "RED", "color": "green"},
        {"word": "GREEN", "color": "blue"},
        {"word": "ORANGE", "color": "yellow"}
    ]
]

MIRROR_TRIALS = [
    [
        {
            "target": [[1,0,0],[1,1,0],[0,1,1]],
            "options": [
                [[0,0,1],[0,1,1],[1,1,0]],
                [[1,0,0],[1,1,0],[0,1,1]],
                [[0,1,1],[1,1,0],[1,0,0]],
                [[1,1,0],[0,1,1],[0,0,1]]
            ]
        },
        {
            "target": [[0,1,0],[1,1,1],[1,0,0]],
            "options": [
                [[0,1,0],[1,1,1],[0,0,1]],
                [[1,1,1],[0,1,0],[1,0,0]],
                [[1,0,0],[1,1,1],[0,1,0]],
                [[0,1,0],[1,1,1],[1,0,0]]
            ]
        },
        {
            "target": [[1,1,0],[0,1,0],[0,1,1]],
            "options": [
                [[0,1,1],[0,1,0],[1,1,0]],
                [[1,1,0],[0,1,0],[0,1,1]],
                [[0,1,0],[1,1,0],[1,1,1]],
                [[1,1,1],[0,1,0],[0,1,1]]
            ]
        },
        {
            "target": [[0,0,1],[1,1,1],[1,0,0]],
            "options": [
                [[1,0,0],[1,1,1],[0,0,1]],
                [[0,0,1],[1,1,1],[1,0,0]],
                [[1,1,1],[1,0,0],[0,0,1]],
                [[0,1,0],[1,1,1],[1,0,1]]
            ]
        },
        {
            "target": [[1,1,1],[1,0,1],[1,0,0]],
            "options": [
                [[0,0,1],[1,0,1],[1,1,1]],
                [[1,1,1],[1,0,1],[1,0,0]],
                [[1,0,0],[1,0,1],[1,1,1]],
                [[1,1,1],[0,1,0],[0,0,1]]
            ]
        }
    ],
    [
        {
            "target": [[0,1,1],[0,1,0],[1,1,0]],
            "options": [
                [[1,1,0],[0,1,0],[0,1,1]],
                [[0,1,1],[0,1,0],[1,1,0]],
                [[1,0,1],[0,1,0],[1,1,0]],
                [[0,1,1],[1,1,0],[0,1,0]]
            ]
        },
        {
            "target": [[1,0,1],[1,1,1],[0,1,0]],
            "options": [
                [[1,0,1],[1,1,1],[0,1,0]],
                [[1,0,1],[1,1,1],[0,1,0]],
                [[1,0,1],[1,1,1],[0,1,0]],
                [[0,1,0],[1,1,1],[1,0,1]]
            ]
        },
        {
            "target": [[1,1,0],[1,0,0],[1,1,1]],
            "options": [
                [[0,1,1],[0,0,1],[1,1,1]],
                [[1,1,0],[1,0,0],[1,1,1]],
                [[1,1,1],[1,0,0],[1,1,0]],
                [[0,0,1],[0,1,1],[1,1,1]]
            ]
        },
        {
            "target": [[0,1,0],[0,1,1],[1,1,0]],
            "options": [
                [[0,1,0],[1,1,0],[0,1,1]],
                [[0,1,0],[0,1,1],[1,1,0]],
                [[1,1,0],[0,1,1],[0,1,0]],
                [[1,0,1],[1,1,0],[0,1,0]]
            ]
        },
        {
            "target": [[1,1,1],[0,1,0],[0,1,0]],
            "options": [
                [[1,1,1],[0,1,0],[0,1,0]],
                [[1,1,1],[0,1,0],[0,1,0]],
                [[1,1,1],[0,1,0],[0,1,0]],
                [[0,1,0],[0,1,0],[1,1,1]]
            ]
        }
    ],
    [
        {
            "target": [[1,0,0],[1,1,1],[0,0,1]],
            "options": [
                [[0,0,1],[1,1,1],[1,0,0]],
                [[1,0,0],[1,1,1],[0,0,1]],
                [[0,0,1],[1,1,1],[0,1,0]],
                [[1,1,1],[0,0,1],[1,0,0]]
            ]
        },
        {
            "target": [[0,1,1],[1,1,0],[0,1,0]],
            "options": [
                [[1,1,0],[0,1,1],[0,1,0]],
                [[0,1,1],[1,1,0],[0,1,0]],
                [[0,1,0],[1,1,0],[0,1,1]],
                [[1,0,1],[0,1,1],[0,1,0]]
            ]
        },
        {
            "target": [[1,1,1],[0,0,1],[1,1,1]],
            "options": [
                [[1,1,1],[1,0,0],[1,1,1]],
                [[1,1,1],[0,0,1],[1,1,1]],
                [[0,0,1],[1,1,1],[0,0,1]],
                [[1,1,1],[1,1,0],[1,1,1]]
            ]
        },
        {
            "target": [[0,1,0],[1,1,1],[0,1,0]],
            "options": [
                [[0,1,0],[1,1,1],[0,1,0]],
                [[1,0,0],[1,1,1],[1,0,0]],
                [[0,1,0],[1,1,1],[0,1,0]],
                [[0,0,1],[1,1,1],[0,0,1]]
            ]
        },
        {
            "target": [[1,0,1],[1,0,1],[1,1,1]],
            "options": [
                [[1,0,1],[1,0,1],[1,1,1]],
                [[1,0,1],[1,0,1],[1,1,1]],
                [[1,1,1],[1,0,1],[1,0,1]],
                [[0,1,0],[0,1,0],[1,1,1]]
            ]
        }
    ]
]

ARITHMETIC_TRIALS = [
    [
        "Start at 1043\n\n1043 - 17 = ?",
        "1026 - 17 = ?",
        "1009 + 28 = ?",
        "1037 - 19 = ?",
        "1018 + 36 = ?"
    ],
    [
        "Start at 2056\n\n2056 - 27 = ?",
        "2029 + 18 = ?",
        "2047 - 29 = ?",
        "2018 + 35 = ?",
        "2053 - 16 = ?"
    ],
    [
        "Start at 3082\n\n3082 - 34 = ?",
        "3048 + 27 = ?",
        "3075 - 38 = ?",
        "3037 + 45 = ?",
        "3082 - 29 = ?"
    ]
]

def get_session_info():
    info = {
        "Subject ID": "SUB001",
        "Session ID": "S01",
        "Operator ID": "OP1",
        "Sensor model": "Generic Bio-Amp",
        "Number of trials": NUM_TRIALS,
    }
    dlg = gui.DlgFromDict(
        dictionary=info,
        title="SAM40 Cognitive Stress - Session Setup",
        order=["Subject ID", "Session ID", "Operator ID", "Sensor model", "Number of trials"],
    )
    if not dlg.OK:
        core.quit()
    return info

def get_acquisition_checklist():
    checklist_passed = False
    while not checklist_passed:
        checklist = {
            f"1. Sampling rate is strictly {SAMPLING_RATE_HZ} Hz": False,
            "2. Sensors are connected & contact quality is verified": False,
            "3. Recording software is actively receiving data": False,
            "4. Participant is seated comfortably and ready": False,
        }
        dlg = gui.DlgFromDict(
            dictionary=checklist,
            title="STRICT MANDATORY CHECKLIST",
            order=list(checklist.keys())
        )
        if not dlg.OK:
            core.quit()
        
        if all(checklist.values()):
            checklist_passed = True
            return checklist
        else:
            error_dlg = gui.Dlg(title="Error")
            error_dlg.addText("You must verify and check ALL boxes to begin the protocol.")
            error_dlg.show()
            if not error_dlg.OK:
                core.quit()

def setup_directories(subject_id, session_id, base_dir="SAM40_DATASET"):
    root = os.path.join(base_dir, "Subject_%s" % subject_id, "Session_%s" % session_id)
    paths = {
        "root": root,
        "Events": os.path.join(root, "Events"),
        "Metadata": os.path.join(root, "Metadata"),
        "Ratings": os.path.join(root, "Ratings"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

class EventLogger:
    EVENT_FIELDS = ["subject_id", "session_id", "trial_id", "event_type", "task", "timestamp", "sample_index"]

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
                name="PsychoPy_SAM40_Markers",
                type="Markers",
                channel_count=1,
                nominal_srate=0,
                channel_format="string",
                source_id="sam40_protocol_%s_%s" % (subject_id, session_id),
            )
            self.outlet = StreamOutlet(info)

    def log(self, event_type, trial_id="", task=""):
        timestamp = round(self.clock.getTime(), 3)
        sample_index = int(round(timestamp * self.fs)) 
        row = {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "trial_id": trial_id,
            "event_type": event_type,
            "task": task,
            "timestamp": timestamp,
            "sample_index": sample_index,
        }
        self._writer.writerow(row)
        self._fh.flush()
        if self.outlet is not None:
            marker_str = "%s|%s|%s|%s" % (event_type, trial_id, task, timestamp)
            self.outlet.push_sample([marker_str])
        logging.exp("EVENT %s" % row)
        return timestamp, sample_index

    def close(self):
        self._fh.close()

class RatingLogger:
    FIELDS = ["subject_id", "session_id", "trial_id", "stroop_stress", "mirror_stress", "arithmetic_stress"]

    def __init__(self, subject_id, session_id, rating_dir):
        fname = "%s_%s_StressRatings.csv" % (subject_id, session_id)
        self.filepath = os.path.join(rating_dir, fname)
        self._fh = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def log(self, subject_id, session_id, trial_id, stroop, mirror, arithmetic):
        row = {
            "subject_id": subject_id, "session_id": session_id,
            "trial_id": trial_id, 
            "stroop_stress": stroop, "mirror_stress": mirror, 
            "arithmetic_stress": arithmetic
        }
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        self._fh.close()

def build_stimuli(win):
    instructions = visual.TextStim(
        win, text="", height=0.04, color="white", wrapWidth=1.4, alignText='center'
    )
    return instructions

def wait_for_key(keys=("space",), allow_escape=True):
    valid = list(keys) + (["escape"] if allow_escape else [])
    pressed = event.waitKeys(keyList=valid)
    if allow_escape and "escape" in pressed:
        core.quit()
    return pressed[0]

def check_escape():
    if "escape" in event.getKeys(["escape"]):
        core.quit()

def run_timed_block(win, instructions, logger, block_name, task_label, duration, display_text, trial_id):
    logger.log(f"{block_name}_START", trial_id, task_label)
    instructions.text = display_text
    clock = core.Clock()
    while clock.getTime() < duration:
        instructions.draw()
        win.flip()
        check_escape()
        core.wait(0.005)
    logger.log(f"{block_name}_END", trial_id, task_label)

def run_stroop_block(win, instructions, logger, trial_index, trial_id):
    logger.log("Stroop_START", trial_id, "Stroop")
    steps = STROOP_TRIALS[(trial_index - 1) % len(STROOP_TRIALS)]
    step_duration = 5.0
    
    orig_height = instructions.height
    orig_pos = instructions.pos
    instructions.height = 0.12
    
    block_clock = core.Clock()
    for step_idx, step_data in enumerate(steps):
        step_start_time = block_clock.getTime()
        logger.log(f"Stroop_Step_{step_idx+1}_START", trial_id, "Stroop")
        
        instructions.text = step_data["word"]
        instructions.color = step_data["color"]
        
        while (block_clock.getTime() - step_start_time) < step_duration:
            instructions.draw()
            win.flip()
            check_escape()
            core.wait(0.005)
            
        logger.log(f"Stroop_Step_{step_idx+1}_END", trial_id, "Stroop")
        
    instructions.height = orig_height
    instructions.color = "white"
    instructions.pos = orig_pos
    logger.log("Stroop_END", trial_id, "Stroop")

def generate_grid_rects(win, matrix, center_x, center_y, block_size=0.03, spacing=0.004, color="yellow"):
    rects = []
    rows = len(matrix)
    cols = len(matrix[0])
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:
                x = center_x + (c - cols / 2.0) * (block_size + spacing) + (block_size / 2.0)
                y = center_y + ((rows / 2.0) - r) * (block_size + spacing) - (block_size / 2.0)
                rect = visual.Rect(win, width=block_size, height=block_size, fillColor=color, lineColor="darkgrey", lineWidth=1)
                rect.pos = (x, y)
                rects.append(rect)
    return rects

def run_mirror_block(win, instructions, logger, trial_index, trial_id):
    logger.log("Mirror_START", trial_id, "Mirror")
    steps = MIRROR_TRIALS[(trial_index - 1) % len(MIRROR_TRIALS)]
    step_duration = 5.0
    
    for step_idx, step_data in enumerate(steps):
        logger.log(f"Mirror_Step_{step_idx+1}_START", trial_id, "Mirror")
        
        target_matrix = step_data["target"]
        option_matrices = step_data["options"]
        
        title_stim = visual.TextStim(win, text=f"Step {step_idx+1}: Find the reflection", height=0.035, color="white", pos=(0, 0.38))
        target_label = visual.TextStim(win, text="Target Shape", height=0.03, color="lightgray", pos=(0, 0.28))
        prompt_stim = visual.TextStim(win, text="Which shape is the reflected version of the target? Press A, B, C, or D", height=0.03, color="white", pos=(0, -0.02))
        
        target_box = visual.Rect(win, width=0.2, height=0.2, pos=(0, 0.14), lineColor="white", fillColor=None, lineWidth=1.5)
        target_rects = generate_grid_rects(win, target_matrix, center_x=0, center_y=0.14, block_size=0.035, color="yellow")
        
        opt_x_coords = [-0.45, -0.15, 0.15, 0.45]
        opt_labels = ["A", "B", "C", "D"]
        
        option_boxes = []
        all_option_rects = []
        label_stims = []
        
        for i, opt_mat in enumerate(option_matrices):
            bx = opt_x_coords[i]
            by = -0.18
            lbl = visual.TextStim(win, text=opt_labels[i], height=0.03, color="yellow", pos=(bx, -0.07))
            label_stims.append(lbl)
            
            box = visual.Rect(win, width=0.22, height=0.18, pos=(bx, by), lineColor="white", fillColor=None, lineWidth=1.2)
            option_boxes.append(box)
            
            opt_rects = generate_grid_rects(win, opt_mat, center_x=bx, center_y=by, block_size=0.025, color="#9370DB")
            all_option_rects.extend(opt_rects)
        
        step_clock = core.Clock()
        while step_clock.getTime() < step_duration:
            title_stim.draw()
            target_label.draw()
            target_box.draw()
            for r in target_rects:
                r.draw()
                
            prompt_stim.draw()
            
            for box in option_boxes:
                box.draw()
            for lbl in label_stims:
                lbl.draw()
            for r in all_option_rects:
                r.draw()
                
            win.flip()
            
            keys = event.getKeys(keyList=['a', 'b', 'c', 'd', '1', '2', '3', '4', 'escape'])
            if keys:
                if 'escape' in keys:
                    core.quit()
                step_answer = keys[0]
                logger.log(f"Mirror_Step_{step_idx+1}_CHOICE_{step_answer.upper()}", trial_id, "Mirror")
                break
            
            check_escape()
            core.wait(0.005)
            
        logger.log(f"Mirror_Step_{step_idx+1}_END", trial_id, "Mirror")
        
    logger.log("Mirror_END", trial_id, "Mirror")

def run_arithmetic_block(win, instructions, logger, trial_index, trial_id):
    logger.log("Arithmetic_START", trial_id, "Math")
    steps = ARITHMETIC_TRIALS[(trial_index - 1) % len(ARITHMETIC_TRIALS)]
    step_duration = 5.0
    
    block_clock = core.Clock()
    for step_idx, step_text in enumerate(steps):
        step_start_time = block_clock.getTime()
        logger.log(f"Arithmetic_Step_{step_idx+1}_START", trial_id, "Math")
        
        while (block_clock.getTime() - step_start_time) < step_duration:
            instructions.text = step_text
            instructions.draw()
            win.flip()
            check_escape()
            core.wait(0.005)
            
        logger.log(f"Arithmetic_Step_{step_idx+1}_END", trial_id, "Math")
        
    logger.log("Arithmetic_END", trial_id, "Math")

def run_trial(win, instructions, trial_id, logger, trial_idx):
    logger.log("TRIAL_START", trial_id)
    
    run_timed_block(win, instructions, logger, "relax", "Baseline", DUR_RELAX_LONG, "Relaxation\n\nPlease sit still and relax.", trial_id)
    
    run_timed_block(win, instructions, logger, "instructions_stroop", "Stroop", DUR_INSTRUCTIONS, "Instructions: Stroop Test\n\nName the color of the ink in which the word is written, ignoring the semantic meaning.", trial_id)
    run_stroop_block(win, instructions, logger, trial_idx, trial_id)
    
    run_timed_block(win, instructions, logger, "relax_short_1", "Rest", DUR_RELAX_SHORT, "Relax", trial_id)
    
    run_timed_block(win, instructions, logger, "instructions_mirror", "Mirror", DUR_INSTRUCTIONS, "Instructions: Mirror Image Recognition\n\nIdentify the reflected shape option (A, B, C, or D) by pressing the corresponding key.", trial_id)
    run_mirror_block(win, instructions, logger, trial_idx, trial_id)
    
    run_timed_block(win, instructions, logger, "relax_short_2", "Rest", DUR_RELAX_SHORT, "Relax", trial_id)
    
    run_timed_block(win, instructions, logger, "instructions_math", "Math", DUR_INSTRUCTIONS, "Instructions: Arithmetic Problem Solving\n\nEvaluate the arithmetic problems sequentially as fast as possible.", trial_id)
    run_arithmetic_block(win, instructions, logger, trial_idx, trial_id)
    
    logger.log("TRIAL_END", trial_id)

def collect_stress_rating(win, instructions, task_name):
    instructions.pos = (0, 0.2)
    instructions.text = f'Rate your stress for {task_name}\n(1 = Min, 10 = Max)\n\nClick on the scale to confirm.'
    
    scale = visual.Slider(
        win,
        ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        labels=['1', '10'],
        granularity=1,
        style='rating',
        color='white',
        font='Arial',
        pos=(0, -0.1) 
    )
    
    while not scale.getRating():
        instructions.draw()
        scale.draw()
        win.flip()
        check_escape()
    
    instructions.pos = (0, 0)
    return scale.getRating()

def write_metadata(paths, info, checklist, session_start_iso):
    metadata = {
        "subject_id": info["Subject ID"],
        "session_id": info["Session ID"],
        "date": session_start_iso,
        "operator_id": info["Operator ID"],
        "sensor_model": info["Sensor model"],
        "sampling_frequency_hz": SAMPLING_RATE_HZ,
        "number_of_trials": info["Number of trials"],
        "protocol_timings": {
            "long_relax_s": DUR_RELAX_LONG,
            "short_relax_s": DUR_RELAX_SHORT,
            "task_s": DUR_TASK,
            "instructions_s": DUR_INSTRUCTIONS
        },
        "software_version": SOFTWARE_VERSION,
        "acquisition_checklist": checklist,
    }
    fname = "%s_%s_Metadata.json" % (info["Subject ID"], info["Session ID"])
    fpath = os.path.join(paths["Metadata"], fname)
    with open(fpath, "w") as fh:
        json.dump(metadata, fh, indent=2)
    return fpath

def main():
    info = get_session_info()
    checklist = get_acquisition_checklist() 

    n_trials = int(info["Number of trials"])
    paths = setup_directories(info["Subject ID"], info["Session ID"])

    win = visual.Window(
        size=[1470, 956],
        monitor=my_monitor,    
        fullscr=True,
        color="black",
        units="height",
    )
    
    instructions = build_stimuli(win)

    logger = EventLogger(info["Subject ID"], info["Session ID"], paths["Events"])
    ratings = RatingLogger(info["Subject ID"], info["Session ID"], paths["Ratings"])

    session_start_iso = datetime.datetime.now().isoformat()
    logger.log("SESSION_START")

    if not LSL_AVAILABLE:
        instructions.text = "NOTE: pylsl is not installed.\nEvent markers will be logged to the CSV file only.\n\nPress SPACE to continue."
        instructions.draw()
        win.flip()
        wait_for_key(["space"])

    for t_idx in range(1, n_trials + 1):
        trial_id = "T%02d" % t_idx

        instructions.text = f"Trial {t_idx} of {n_trials}\n\nPress SPACE to begin."
        instructions.draw()
        win.flip()
        wait_for_key(["space"])

        run_trial(win, instructions, trial_id, logger, t_idx)

        stroop_stress = collect_stress_rating(win, instructions, "Stroop Test")
        mirror_stress = collect_stress_rating(win, instructions, "Mirror Image")
        math_stress = collect_stress_rating(win, instructions, "Arithmetic")

        ratings.log(info["Subject ID"], info["Session ID"], trial_id, stroop_stress, mirror_stress, math_stress)

    logger.log("SESSION_END")

    meta_path = write_metadata(paths, info, checklist, session_start_iso)

    logger.close()
    ratings.close()

    instructions.text = f"Session complete.\n\nEvent file:    {logger.filepath}\nRatings file:  {ratings.filepath}\nMetadata file: {meta_path}\n\nPress SPACE to exit."
    instructions.draw()
    win.flip()
    wait_for_key(["space"])

    win.close()
    core.quit()

if __name__ == "__main__":
    main()