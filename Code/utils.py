import torch
import random
import numpy as np
import itertools


def augment_training_pairs(training_pairs):
    augmented_pairs = []
    
    for pair in training_pairs:
        original_input = np.array(pair['input'])
        original_output = np.array(pair['output'])

        geometric_versions = [
            (original_input, original_output),
            (np.fliplr(original_input), np.fliplr(original_output)),
            (np.flipud(original_input), np.flipud(original_output))
        ]

        for input_grid, output_grid in geometric_versions:
            augmented_pairs.append({'input': input_grid.tolist(), 'output': output_grid.tolist()})
            
            color_augmented_pairs = _generate_color_permutations(input_grid, output_grid)
            augmented_pairs.extend(color_augmented_pairs)

    return augmented_pairs


def _generate_color_permutations(input_grid, output_grid):
    permutated_pairs = []
    
    original_colors = np.unique(np.concatenate((input_grid.flatten(), output_grid.flatten())))
    original_colors = [c for c in original_colors if c != 0]
    num_original_colors = len(original_colors)

    if num_original_colors == 0:
        return []

    all_possible_colors = set(range(1, 10))
    available_colors = list(all_possible_colors - set(original_colors))

    if len(available_colors) < num_original_colors:
        return []

    num_augmentations_to_generate = 2 * num_original_colors

    all_target_color_sets = list(itertools.combinations(available_colors, num_original_colors))
    
    num_possible_unique_sets = len(all_target_color_sets)
    num_to_sample = min(num_augmentations_to_generate, num_possible_unique_sets)

    chosen_color_sets = random.sample(all_target_color_sets, num_to_sample)

    for new_colors_tuple in chosen_color_sets:
        color_map = {original: new for original, new in zip(original_colors, new_colors_tuple)}

        new_input = input_grid.copy()
        new_output = output_grid.copy()

        for original_color, new_color in color_map.items():
            new_input[new_input == original_color] = new_color
            new_output[new_output == original_color] = new_color
        
        permutated_pairs.append({'input': new_input.tolist(), 'output': new_output.tolist()})
        
    return permutated_pairs


def format_grid_to_string(grid):
    if isinstance(grid, torch.Tensor):
        grid = grid.int().tolist()
    return '|'.join([','.join(map(str, row)) for row in grid])


def calculate_grid_dot_product(grid):
    if not grid or not grid[0]:
        return 0
    height = len(grid)
    width = len(grid[0])
    return height * width