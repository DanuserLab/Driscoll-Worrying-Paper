# Driscoll-Worrying-Paper
This bundle contains code used in **Proteolysis-free amoeboid migration through crowded environment via bleb-driven worrying**, *Developmental Cell*, 2024, written by Meghan K. Driscoll, Erik S. Welf, Andrew Weems, Etai Sapoznik, Felix Zhou, Vasanth S. Murali, Juan Manuel Garcia-Arcos, Minna Roh-Johnson, Matthieu Piel, Kevin M. Dean, Reto Fiolka, [Gaudenz Danuser](https://www.danuserlab-utsw.org/). Additional information can be found in the Methods section of this paper.

**2D Bleb tracking and analysis** – Python code used to detect and track blebs frame-to-frame, and to analyze bleb size statistics before and after photoactivation. 

The code is organized into steps: 

Step1_Detection_and_Tracking: two scripts: bleb_tracking.py module file containing the reusable scripts for tracking and ‘step1_detect-blebs_and_track.py’, the main script for extracting the bleb timeseries for analysis.

Step2_Before_After_bleb_stats: one script to compute the before and after bleb sizes and perform statistical test, another script to coplot the conditions. 

Step3_Time_Correlation_Analysis: one script to compute the cross-correlation of the timeseries of bleb area and Arp2/3, another script to coplot.

The code is shared for transparent documentation of the analyses performed in the manuscript by Driscoll et al.
