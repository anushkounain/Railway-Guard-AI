# Data Directory

This directory stores videos for analysis, zone configuration files, and the local SQLite database.

## Contents

- `zones.json`: Configured polygon zones (track intrusion zones, level crossing zones).
- `railway_safety.db`: Persistent SQLite database storing events, violations, and session history.
- `uploads/`: Temporary or uploaded video files awaiting or undergoing inference.

## Video Storage Note

Raw video files (`*.mp4`, `*.avi`, `*.mov`) are excluded from Git commits via `.gitignore`.
Uploaded surveillance clips remain local to your machine.
