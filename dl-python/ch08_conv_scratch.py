# %%
import glob
import os
import random
import shutil
import zipfile
from pathlib import Path

import kagglehub

# %%

download_path = Path(kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset"))


# %%
def make_small_dataset(n_split: list[int], download_path: Path, target_path: Path = Path(".")):
    for category in ("Cat", "Dog"):
        category_path = download_path / "PetImages" / category
        samples = set(category_path.glob("*.jpg"))

        for split, n in zip(["train", "validation", "test"], n_split):
            split_samples = random.sample(list(samples), n)

            (target_path / split / category).mkdir(parents=True)
            for file in split_samples:
                shutil.copy(src=category_path / file, dst=target_path / split / category)

            samples -= set(split_samples)


# %%
make_small_dataset([1000, 500, 1000], download_path, Path("ch08-cats-vs-dogs"))

# %%
