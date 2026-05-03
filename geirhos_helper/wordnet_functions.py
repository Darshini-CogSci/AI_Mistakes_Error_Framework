
import numpy as np
from shutil import copyfile
import os
import linecache as lc

def get_filenames_of_category(category, image_labels_path, categories):
    """Return a list of filenames of all images belonging to a category.

    category - a string specifying a (perhaps broad) category
    image_labels_path - a filepath to a file with all image labels,
                        formatted in the ilsvrc2012 format
    categories - a list of all categories of the dataset. The order of
                 categories has to be the same as used for the labelling.

    """
  
    # get indices of all subcategories that belong to the category
    subcategories_list = []
    counter = 0
    for c in categories:
        if is_hypernym(c, category):
            subcategories_list.append(counter)
        counter += 1


    image_list = []
    with open(image_labels_path) as labels_file:
        for line in labels_file:
            image_name, image_label = line.split(" ")

            if int(image_label) in subcategories_list:
                image_list.append(image_name)

    return image_list


def hypernyms_in_ilsvrc2012_categories(entity):
    """Return all hypernyms of categories.txt for a given entity.

    entity - a string, e.g. "furniture"
  
    Returns the children of the entity, e.g. "bed" and "chair" if there were
    both a "bed" and a "chair" in categories.txt (the imagenet categories).
    If the entity itself is contained, it will be returned as well.
    """

    return get_hypernyms("categories.txt", entity)

    
def get_hypernyms(categories_file, entity):
    """Return all hypernyms of categories for a given entity.

    entity - a string, e.g. "furniture"

    Returns the children of the entity, e.g. "bed" and "chair" if there were
    both a "bed" and a "chair" in the categories.
    If the entity itself is contained, it will be returned as well.
    """

    hypers = []
    with open(categories_file) as f:
        for line in f:
            category = get_category_from_line(line)
            cat_synset = wn.synsets(category)[0]
            if is_hypernym(category, entity):
                hypers.append(category)

    return hypers


def get_ilsvrc2012_training_WNID(entity):
    """Return a WNID for each hypernym of entity.

    entity - a string, e.g. "furniture"

    Returns the WNIDs of the children of the entity,
    e.g. "bed" and "chair" if there were
    both a "bed" and a "chair" in the ilsvrc2012 categories.
    If the entity itself is contained, it will be returned as well.
    """

    results = []

    hypernyms = hypernyms_in_ilsvrc2012_categories(entity)
    
    for hyper in hypernyms:

        with open("WNID_synsets_mapping.txt") as f:
            for line in f:
                category = get_category_from_line(line)

                if category == hyper:
                    print(line[:9])
                    results.append(line[:9])

    return results


def num_hypernyms_in_ilsvrc2012(entity):
    """Return number of hypernyms in the ilsvrc2012 categories for entity."""

    return len(hypernyms_in_ilsvrc2012_categories(entity))


def get_ilsvrc2012_categories():
    """Return the first item of each synset of the ilsvrc2012 categories."""

    categories = []

    with open("categories.txt") as f:
        for line in f:
           categories.append(get_category_from_line(line))

    return categories


def get_category_from_line(line):
    """Return the category without anything else from categories.txt"""

    category = line.split(",")[0][10:]
    category = category.replace(" ", "_")
    category = category.replace("\n", "")
    return category           


def get_WNID_from_index(index):
    """Return WNID given an index of categories.txt"""
    assert(index >= 0 and index < 1000), "index needs to be within [0, 999]"

    file_path = r"C:\Users\Darshini\Downloads\texture-vs-shape-master\code\helper\categories.txt"
    assert(os.path.exists(file_path)), "path to categories.txt wrong!"
    line = lc.getline(file_path, index+1)
    return line.split(" ")[0]

import json

def load_imagenet_class_idx():
    """Loads the ImageNet class index mapping from a local file."""
    try:
        # The name of your JSON file containing the mapping
        with open('imagenet_class_index.json') as f:
            imagenet_class_index = json.load(f)
        return imagenet_class_index
    except FileNotFoundError:
        print("Error: 'imagenet_class_index.json' not found. Please download it.")
        return None

def get_imagenet_label_from_index(index):
    """
    Retrieves the human-readable label for a given ImageNet index by reading
    the categories.txt file.
    
    Args:
        index (int): The ImageNet index (0-999).
        
    Returns:
        str: The human-readable label for the index.
    """
    file_path = r"C:\Users\Darshini\Downloads\texture-vs-shape-master\code\helper\categories.txt"
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    # The format is 'WNID label, other labels...'
                    # The label starts after the 10th character (after WNID and a space)
                    # and ends at the first comma.
                    label = line[10:].split(',', 1)[0].strip().replace('_', ' ')
                    return label
        return 'Unknown Label'
    except FileNotFoundError:
        return 'File Not Found'