# Particle Filter Tracker - motion model comparison results

python3 compare_trackers.py --workspace_path "/media/samuponz/03DD6CF02B183CCE/FRI - Ljubljana/ATCV - Advanced Topics in Computer Vision/Projects/workspace-dir/" --trackers pf_tracker_rw_N_100 pf_tracker_ncv_N_100 pf_tracker_nca_N_100
------------------------------------
Results for tracker: PFTracker_rw_N_100
  Average overlap: 0.45
  Total failures: 50.0
  Average speed: 68.38 FPS
------------------------------------
Performing evaluation for tracker: PFTracker_ncv_N_100
------------------------------------
Results for tracker: PFTracker_ncv_N_100
  Average overlap: 0.46
  Total failures: 54.0
  Average speed: 75.73 FPS
------------------------------------
------------------------------------
Results for tracker: PFTracker_nca_N_100
  Average overlap: 0.47
  Total failures: 98.0
  Average speed: 68.13 FPS
------------------------------------

# Particle Filter Tracker - number of particles comparison results

python3 compare_trackers.py --workspace_path "/media/samuponz/03DD6CF02B183CCE/FRI - Ljubljana/ATCV - Advanced Topics in Computer Vision/Projects/workspace-dir/" --trackers pf_tracker_ncv_N_50 pf_tracker_ncv_N_100 pf_tracker_ncv_N_150 pf_tracker_ncv_N_200 pf_tracker_ncv_N_250 pf_tracker_ncv_N_300

------------------------------------
Results for tracker: PFTracker_ncv_N_50
  Average overlap: 0.45
  Total failures: 50.0
  Average speed: 143.66 FPS
------------------------------------
------------------------------------
Results for tracker: PFTracker_ncv_N_100
  Average overlap: 0.46
  Total failures: 54.0
  Average speed: 75.73 FPS
------------------------------------
------------------------------------
Results for tracker: PFTracker_ncv_N_150
  Average overlap: 0.46
  Total failures: 52.0
  Average speed: 52.62 FPS
------------------------------------
------------------------------------
Results for tracker: PFTracker_ncv_N_200
  Average overlap: 0.47
  Total failures: 45.0
  Average speed: 35.40 FPS
------------------------------------
------------------------------------
Results for tracker: PFTracker_ncv_N_250
  Average overlap: 0.46
  Total failures: 52.0
  Average speed: 33.32 FPS
------------------------------------
Performing evaluation for tracker: PFTracker_ncv_N_300
------------------------------------
Results for tracker: PFTracker_ncv_N_300
  Average overlap: 0.47
  Total failures: 51.0
  Average speed: 27.95 FPS
------------------------------------
