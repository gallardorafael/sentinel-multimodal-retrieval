import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class FlickrImage:
    """A dataclass representing a Flickr image and its captions.

    Attributes:
        image_path (str): The path to the image file.
        captions (List[str]): A list of captions for the image.
        best_caption (str): The best caption for the image.
    """

    image_path: Path
    captions: List[str]
    best_caption: str

    def __str__(self):
        """Returns a string representation of the FlickrImage object."""
        return f"{self.image_path}: {self.captions}"


class FlickrDataset:
    """An iterator dataset class for handling Flickr image-caption pairs.

    Attributes:
        root_path (str): The root directory path where images and captions are stored.
        image_paths (list): A list of paths to the image files.
        captions_dict (defaultdict): A dictionary mapping image filenames to their corresponding captions.
        index (int): The current index for iteration.
    """

    def __init__(self, root_path: str):
        """Initializes the FlickrDataset object.

        Args:
            root_path (str): The root directory path where images and captions are stored. Note that
            the captions are stored in a file named "captions.txt", and the images are stored in a
            folder named "images".
        """
        self.root_path = Path(root_path)
        self.image_paths = []
        self.captions_dict = defaultdict(list)
        self.index = 0
        self._load_data()

    def _load_data(self):
        """Loads the image paths and captions from the root path."""
        # getting all the image paths from the root_path
        self.image_paths = list(self.root_path.glob("**/*.jpg"))

        # loading the captions
        with open(self.root_path / "captions.txt", "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            rows = [row for row in csv_reader]

        for row in rows:
            self.captions_dict[row[0]].append(row[1])

    def __iter__(self):
        """Resets the index for iteration."""
        self.index = 0
        return self

    def __next__(self):
        """Returns the next FlickrImage object in the dataset.

        Returns:
            FlickrImage: A FlickrImage object representing an image and its captions.
        """
        if self.index >= len(self.image_paths):
            raise StopIteration
        image_path = self.image_paths[self.index]
        captions = self.captions_dict[image_path.name]
        longer_caption = max(captions, key=len)
        image_obj = FlickrImage(
            image_path=image_path.as_posix(),
            captions=captions,
            best_caption=longer_caption,
        )
        self.index += 1
        return image_obj

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_paths)
