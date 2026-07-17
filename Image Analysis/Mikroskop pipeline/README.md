## Erklärung der Ordner Struktur 
data/ – alles, was mit euren Bildern/Daten zu tun hat

data/raw/
Original-Mikroskopbilder
data/processed/
Bilder nach Preprocessing (Denoising, Kontrastverbesserung, Hintergrundsubtraktion etc. – Step 1-2 aus eurer Pipeline) --> Input für die Segmentierung. 
data/masks_cells/
Ergebnisse der Zellsegmentierung 
data/masks_gv/
manuell erstellten Ground-Truth-Annotationen für Gasvesikel --> napari/ fiji
data/splits/
Textdateien, die festhalten, welche Bilder zu Training, Validierung und Test gehören 


notebooks/ – zum ausprobieren, explorativer Code
01_cellsam_baseline.ipynb: CellSAM zum ersten Mal auf Bildern testen, Ergebnisse anschauen
02_annotation_check.ipynb: GV-Annotationen konsistent/sinnvoll? (z.B. Overlap zwischen zwei Annotierenden)
03_finetuning.ipynb: Trainings-/Fine-Tuning-Lauf

src/ – wiederverwendbarer Code

preprocessing.py: Denoising, Kontrastfunktionen
segmentation.py: CellSAM-Aufrufe, Wrapper-Funktionen
annotation_utils.py: Hilfsfunktionen zum Laden/Konvertieren eurer Annotationen (z.B. napari-Masken → Trainingsformat)
metrics.py: Precision/Recall/F1/IoU-Berechnungen, um Modelle objektiv zu vergleichen

models/ – trainierte Gewichte
Gespeicherte Checkpoints nach dem Fine-Tuning.

results/ – Output für Menschen
Fertige Plots, etc.

project.toml + uv.lock --> packages, dependencies, für reproduzierbarkeit in venv / .gitignore --> was wird nicht gepushed / README.md --> Erklärung
