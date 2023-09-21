### überall einzufügen, wo ein VRAM-Check gemacht werden soll. Die Zeilen gc.collect und torch.cuda.empty_cache() sind vermutlich nicht zwingend nötig...

print("NAME:", torch.cuda.max_memory_allocated(device="cuda:0") / 1024 / 1024)
# alternativ: Zeitstempel + name + Wert in Datei-schreiben...
torch.cuda.reset_peak_memory_stats()
gc.collect()
torch.cuda.empty_cache()