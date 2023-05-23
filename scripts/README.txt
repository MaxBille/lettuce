Skripte zum Durchführen einer Simulation der 2D Zylinderumströmung:

1. 2d_cylinder_simulation.py
    - simuliert einen 2D cylinder flow
    - nimmt als direkte Kommandozeilen-Argumente: --re Reynoldszahl --n_steps Zeitschrittzahl --gpd GridpointsPerDiameter --dpy DiametersPerDomainwidthY --t_target PUZeitZuSimulieren --collision Kollisionsoperator; Jeweils mit default Werten (siehe Skript)
    - innerhalb des Skripts können Ausgabepfad, Ausgabename und weitere Parameter gesetzt werden
    - weitere Kommentare: siehe Skript/Code

2. cluster_job_script.sh
    - startet einen Cluster-Job und führt dort mit den gewünschten Parametern das Skript "2d_cylinder_simulation.py" aus
    - Jobname kann im Skript definier werden
    - nimmt als Argumente: Reynoldszahl, GPD, DpY, T_Target. Weitere Argumente (, die auch 2d_cylinder_simulation.py unterstützt) können vopm Nutzer hinzugefügt werden
    - Benennung der direkten stdout-Ergebnisse des Skriptes "2d_cylinder_simulation.py" mit Job-ID, Job-Name und Parmetern.
    - weitere Simulationseinstellungen, Benennung der Ergebnisse etc. in "2d_cylinder_simulation.py" (s.o.)
