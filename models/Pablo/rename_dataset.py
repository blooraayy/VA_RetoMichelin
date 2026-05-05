# =============================================================================
# rename_dataset.py
# Renombra las imágenes del dataset de Roboflow a un formato limpio
# y actualiza el _annotations.coco.json para que los nombres coincidan.
#
# Uso: python rename_dataset.py
# Ejecutar desde la raíz del proyecto.
# =============================================================================

import os
import json
import re
import shutil

IMAGE_DIR   = "data/images"
LABEL_FILE  = "data/labels/_annotations.coco.json"

def clean_name(filename: str) -> str:
    """Convierte 'Multimedia (42)_jpg.rf.HASH.jpg' -> 'Multimedia_42.jpg'"""
    # Quitar el sufijo .rf.HASH
    name = re.sub(r'_jpg\.rf\.[a-zA-Z0-9]+', '', filename)
    # Quitar extensión para trabajar con el stem
    stem = os.path.splitext(name)[0]
    # Reemplazar espacios y paréntesis: "Multimedia (42)" -> "Multimedia_42"
    stem = stem.replace(" (", "_").replace(")", "")
    stem = stem.replace(" ", "_")
    return stem + ".jpg"


def main():
    # Cargar JSON
    with open(LABEL_FILE, "r") as f:
        coco = json.load(f)

    print(f"Imágenes en JSON: {len(coco['images'])}")

    # Construir mapa old_name -> new_name
    rename_map = {}
    for img in coco["images"]:
        old_name = img["file_name"]
        new_name = clean_name(old_name)
        rename_map[old_name] = new_name

    # Mostrar preview
    print("\nPreview de renombrado:")
    for old, new in sorted(rename_map.items()):
        print(f"  {old}  ->  {new}")

    confirm = input("\n¿Continuar? (s/n): ").strip().lower()
    if confirm != "s":
        print("Cancelado.")
        return

    # Renombrar ficheros
    renamed = 0
    not_found = 0
    for old_name, new_name in rename_map.items():
        old_path = os.path.join(IMAGE_DIR, old_name)
        new_path = os.path.join(IMAGE_DIR, new_name)

        if os.path.isfile(old_path):
            os.rename(old_path, new_path)
            print(f"  Renombrado: {old_name} -> {new_name}")
            renamed += 1
        else:
            print(f"  NO ENCONTRADO: {old_path}")
            not_found += 1

    # Actualizar JSON
    for img in coco["images"]:
        old_name = img["file_name"]
        if old_name in rename_map:
            img["file_name"] = rename_map[old_name]

    # Guardar JSON actualizado
    with open(LABEL_FILE, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nFicheros renombrados: {renamed}")
    print(f"Ficheros no encontrados: {not_found}")
    print(f"JSON actualizado: {LABEL_FILE}")
    print("\nListo!")


if __name__ == "__main__":
    main()
