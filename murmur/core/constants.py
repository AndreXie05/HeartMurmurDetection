"""Facts of the task and the dataset that every layer agrees on."""
from __future__ import annotations

CLASSES = ("Present", "Unknown", "Absent")
# Weights of the PhysioNet 2022 murmur metric (project spec §3.4).
CLASS_WEIGHTS = {"Present": 5, "Unknown": 3, "Absent": 1}

SAMPLE_RATE = 4000                       # every CirCor 1.0.3 recording (all 3163 headers say 4000)
LOCATIONS = ("AV", "PV", "TV", "MV", "Phc")

# States of the `.tsv` heart-cycle segmentation; 0 = no state marked.
STATES = {0: "unmarked", 1: "S1", 2: "systole", 3: "S2", 4: "diastole"}
