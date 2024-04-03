# %%
import os
import shutil

OLD_ROOT = "result/"
NEW_ROOT = "result-clean/"

for root, subdirs, files in os.walk(OLD_ROOT):
    for subdir in subdirs:
        os.makedirs(
            os.path.join(root, subdir).replace(OLD_ROOT, NEW_ROOT),
            exist_ok=True,
        )
    for file in files:
        if file.endswith(".pt"):
            continue

        shutil.copyfile(
            os.path.join(root, file),
            os.path.join(root, file).replace(OLD_ROOT, NEW_ROOT),
        )

# %%
