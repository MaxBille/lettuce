Skripte zum Durchführen einer Simulation der 2D Zylinderumströmung:

1. 2d_cylinder_simulation.py
    - simuliert einen 2D cylinder flow
    - nimmt als direkte Kommandozeilen-Argumente: --re Reynoldszahl --n_steps Zeitschrittzahl --gpd GridpointsPerDiameter --dpy DiametersPerDomainwidthY --t_target PUZeitZuSimulieren --collision Kollisionsoperator; Jeweils mit default Werten (siehe Skript)
    - innerhalb des Skripts können Ausgabepfad, Ausgabename und weitere Parameter gesetzt werden
    - weitere Kommentare: siehe Skript/Code

2. cluster_job_script.sh
    - startet einen Cluster-Job und führt dort mit den gewünschten Parametern das Skript "2d_cylinder_simulation.py" aus
    - Jobname kann im Skript definiert werden
    - nimmt als Argumente: Reynoldszahl, GPD, DpY, T_Target. Weitere Argumente (, die auch 2d_cylinder_simulation.py unterstützt) können vom Nutzer hinzugefügt werden
    - Benennung der direkten stdout-Ergebnisse des Skriptes "2d_cylinder_simulation.py" mit Job-ID, Job-Name und Parametern.
    - weitere Simulationseinstellungen, Benennung der Ergebnisse etc. in "2d_cylinder_simulation.py" (s.o.)

Skript zum Plotting des Diskretisierungseinflusses auf die Rundheit des Kreises:

roudness.py
    - plottet die relativen Radien (Abstand eines Gitterpunktes auf dem diskretisierten Kreisumfang zum Zentrum in Relation zum vorgegebenen Radius in LU)
        a) als Histogramm: relativer Anteil verschiedener relativer Radien (zum direkten Vergleich verschiedener Auflösungen)
        b) als Plot: minimaler, maximaler, mittlerer relativer Radius, in Abhängigkeit der Auflösung in GPD (Vergleich vieler Auflösungen)
    - Darstellung jeweils einmal mit jedem Gitterpunkt einfach gezählt und einmal jedem Gitterpunkt entsprechend seiner Anzahl an Links zu Fluidknoten gezählt.