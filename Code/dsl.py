import torch
from collections import deque, Counter
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import re

GRID_DTYPE = torch.bfloat16


class DSLExecutor:
    def __init__(self):
        self.operations = {
            'find_all_objects_by_color_connectivity': self._find_all_objects_by_color_connectivity,
            'select_all_found_objects': self._select_all_found_objects,
            'clear_current_selection': self._clear_current_selection,
            'filter_selection_by_color': self._filter_selection_by_color,
            'filter_selection_by_exact_size': self._filter_selection_by_exact_size,
            'filter_selection_by_size_greater_than': self._filter_selection_by_size_greater_than,
            'filter_selection_by_size_less_than': self._filter_selection_by_size_less_than,
            'filter_selection_by_bounding_box': self._filter_selection_by_bounding_box,
            'filter_selection_by_touching_grid_border': self._filter_selection_by_touching_grid_border,
            'filter_selection_by_not_touching_grid_border': self._filter_selection_by_not_touching_grid_border,
            'filter_selection_by_row_overlap': self._filter_selection_by_row_overlap,
            'filter_selection_by_column_overlap': self._filter_selection_by_column_overlap,
            'select_largest_objects_from_selection': self._select_largest_objects_from_selection,
            'select_smallest_objects_from_selection': self._select_smallest_objects_from_selection,
            'select_object_from_selection_by_size_rank': self._select_object_from_selection_by_size_rank,
            'group_selection_by_color': self._group_selection_by_color,
            'move_selected_objects_by_delta': self._move_selected_objects_by_delta,
            'move_selected_objects_until_contact': self._move_selected_objects_until_contact,
            'extend_selected_objects_in_direction': self._extend_selected_objects_in_direction,
            'scale_selected_objects_from_top_left': self._scale_selected_objects_from_top_left,
            'resize_grid_to_dimensions': self._resize_grid_to_dimensions,
            'rotate_selected_objects_around_center': self._rotate_selected_objects_around_center,
            'flip_selected_objects_around_center': self._flip_selected_objects_around_center,
            'recolor_selected_objects': self._recolor_selected_objects,
            'copy_selected_objects_in_place': self._copy_selected_objects_in_place,
            'delete_selected_objects': self._delete_selected_objects,
            'crop_grid_to_selection_bounding_box': self._crop_grid_to_selection_bounding_box,
            'set_grid_background_color': self._set_grid_background_color,
            'fill_entire_grid_with_color': self._fill_entire_grid_with_color,
            'repeat_sub_operation': self._repeat_sub_operation,
            'apply_sub_operation_to_each_selected_object': self._apply_sub_operation_to_each_selected_object,
            'apply_color_pattern_to_grid': self._apply_color_pattern_to_grid,
            'save_current_selection_to_memory': self._save_current_selection_to_memory,
            'load_selection_from_memory': self._load_selection_from_memory,
            'clear_saved_selection_from_memory': self._clear_saved_selection_from_memory,
            'count_selected_objects_to_variable': self._count_selected_objects_to_variable,
            'get_color_of_first_selected_object_to_variable': self._get_color_of_first_selected_object_to_variable,
            'get_size_of_first_selected_object_to_variable': self._get_size_of_first_selected_object_to_variable,
            'get_position_of_first_selected_object_to_variables': self._get_position_of_first_selected_object_to_variables,
            'get_grid_dimensions_to_variables': self._get_grid_dimensions_to_variables,
            'set_variable_value': self._set_variable_value,
            'add_to_variable': self._add_to_variable,
            'subtract_from_variable': self._subtract_from_variable,
            'multiply_variable_by': self._multiply_variable_by,
            'divide_variable_by': self._divide_variable_by,
            'modulo_variable_by': self._modulo_variable_by,
            'intersect_current_selection_with_saved': self._intersect_current_selection_with_saved,
            'union_current_selection_with_saved': self._union_current_selection_with_saved,
            'xor_current_selection_with_saved': self._xor_current_selection_with_saved,
            'invert_current_selection': self._invert_current_selection,
            'group_selected_objects_by_connectivity': self._group_selected_objects_by_connectivity,
            'get_center_of_selection_to_variables': self._get_center_of_selection_to_variables,
            'align_selected_objects': self._align_selected_objects,
            'distribute_selected_objects': self._distribute_selected_objects,
            'map_all_grid_colors': self._map_all_grid_colors,
            'project_selection_onto_axis': self._project_selection_onto_axis,
            'copy_selection_and_apply_sub_operation': self._copy_selection_and_apply_sub_operation,
            'mirror_selection_across_grid_axis': self._mirror_selection_across_grid_axis,
            'delete_pixels_of_selection': self._delete_pixels_of_selection,
            'compose_selected_objects_into_new_grid': self._compose_selected_objects_into_new_grid,
            'get_property_of_first_selected_object_to_variable': self._get_property_of_first_selected_object_to_variable,
            'execute_sub_operation_if_variable_equals': self._execute_sub_operation_if_variable_equals,
            'hollow_selected_objects': self._hollow_selected_objects,
            'rotate_selected_objects_90_degrees': self._rotate_selected_objects_90_degrees,
            'rotate_selected_objects_180_degrees': self._rotate_selected_objects_180_degrees,
            'rotate_selected_objects_270_degrees': self._rotate_selected_objects_270_degrees,
            'filter_selection_by_symmetry': self._filter_selection_by_symmetry,
            'get_selection_bounding_box_to_variables': self._get_selection_bounding_box_to_variables,
            'count_unique_colors_in_selection_to_variable': self._count_unique_colors_in_selection_to_variable,
            'get_most_frequent_color_in_selection_to_variable': self._get_most_frequent_color_in_selection_to_variable,
            'get_least_frequent_color_in_selection_to_variable': self._get_least_frequent_color_in_selection_to_variable,
            'flood_fill_from_selection': self._flood_fill_from_selection,
            'draw_line_between_centers_of_two_selected_objects': self._draw_line_between_centers_of_two_selected_objects,
            'draw_border_around_selection': self._draw_border_around_selection,
            'invert_colors_in_selection': self._invert_colors_in_selection,
            'shift_colors_in_selection': self._shift_colors_in_selection,
            'apply_kernel_from_variable_to_selection': self._apply_kernel_from_variable_to_selection,
            'filter_selection_by_neighbor_count': self._filter_selection_by_neighbor_count,
            'select_holes_within_selection': self._select_holes_within_selection,
            'fill_holes_within_selection': self._fill_holes_within_selection,
            'filter_selection_by_aspect_ratio': self._filter_selection_by_aspect_ratio,
            'stack_selected_objects_vertically': self._stack_selected_objects_vertically,
            'stack_selected_objects_horizontally': self._stack_selected_objects_horizontally,
            'wrap_grid_by_delta': self._wrap_grid_by_delta,
            'get_pixel_color_to_variable': self._get_pixel_color_to_variable,
            'set_pixel_color': self._set_pixel_color,
            'filter_selection_by_convexity': self._filter_selection_by_convexity,
            'transform_selection_to_convex_hull': self._transform_selection_to_convex_hull,
            'mask_selection_with_saved_selection': self._mask_selection_with_saved_selection,
            'count_islands_of_color_to_variable': self._count_islands_of_color_to_variable,
            'sort_selection_by_position': self._sort_selection_by_position,
            'shear_selection': self._shear_selection,
            'split_grid_at_position_to_memory': self._split_grid_at_position_to_memory,
            'merge_grids_from_memory': self._merge_grids_from_memory,
            'filter_selection_by_top_left_parity': self._filter_selection_by_top_left_parity,
            'keep_only_diagonal_pixels_of_selection': self._keep_only_diagonal_pixels_of_selection,
            'move_selection_to_grid_border': self._move_selection_to_grid_border,
            'swap_positions_of_two_selected_objects': self._swap_positions_of_two_selected_objects,
            'crop_grid_to_all_content': self._crop_grid_to_all_content,
            'pad_grid_with_color': self._pad_grid_with_color,
            'count_all_colors_in_grid_to_variables': self._count_all_colors_in_grid_to_variables,
            'select_objects_contained_within_others': self._select_objects_contained_within_others,
            'move_selection_until_collision': self._move_selection_until_collision,
            'get_grid_color_palette_to_variable': self._get_grid_color_palette_to_variable,
            'replace_color_within_selection': self._replace_color_within_selection,
            'create_object_from_selection_bounding_box': self._create_object_from_selection_bounding_box,
            'select_background_as_object': self._select_background_as_object,
            'filter_selection_by_hole_count': self._filter_selection_by_hole_count,
            'get_object_hole_count_to_variable': self._get_object_hole_count_to_variable,
            'union_all_selected_objects': self._union_all_selected_objects,
            'recolor_object_border': self._recolor_object_border,
            'recolor_object_interior': self._recolor_object_interior,
            'smear_selection_in_direction': self._smear_selection_in_direction,
            'create_checkerboard_pattern': self._create_checkerboard_pattern,
            'tile_grid_with_selection': self._tile_grid_with_selection,
            'get_object_perimeter_to_variable': self._get_object_perimeter_to_variable,
            'filter_selection_by_perimeter_length': self._filter_selection_by_perimeter_length,
            'select_objects_in_row': self._select_objects_in_row,
            'select_objects_in_column': self._select_objects_in_column,
            'get_object_shape_as_boolean_mask_to_variable': self._get_object_shape_as_boolean_mask_to_variable,
            'create_line_object': self._create_line_object,
            'move_object_to_absolute_position': self._move_object_to_absolute_position,
            'select_object_at_absolute_position': self._select_object_at_absolute_position,
            'extrude_selection': self._extrude_selection,
            'get_pixel_neighbor_colors_to_variable': self._get_pixel_neighbor_colors_to_variable,
            'count_objects_in_row_to_variable': self._count_objects_in_row_to_variable,
            'count_objects_in_column_to_variable': self._count_objects_in_column_to_variable,
            'split_selection_by_color': self._split_selection_by_color,
            'create_grid_from_selection': self._create_grid_from_selection,
            'overlay_grid_from_variable': self._overlay_grid_from_variable,
            'select_objects_by_shape_template': self._select_objects_by_shape_template,
            'apply_cellular_automaton_rule_to_selection': self._apply_cellular_automaton_rule_to_selection
        }

    def get_operation_names(self):
        return list(self.operations.keys())

    def execute_script(self, script_string, input_grid):
        if not isinstance(input_grid, torch.Tensor):
            grid_tensor = torch.tensor(
                input_grid, dtype=GRID_DTYPE
            )
        else:
            grid_tensor = input_grid.clone().to(GRID_DTYPE)

        state = {
            'input_grid': grid_tensor,
            'current_grid': grid_tensor.clone(),
            'all_objects': [],
            'selected_objects': [],
            'selection_groups': {},
            'variables': {},
            'saved_selections': {}
        }

        script_lines = [
            line.strip() for line in script_string.strip().split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]

        for line in script_lines:
            try:
                state = self._parse_and_execute_line(line, state)
            except Exception:
                return None

        return state['current_grid']

    def _resolve_argument(self, argument, state):
        if isinstance(argument, str) and argument.startswith('var:'):
            variable_name = argument.split(':', 1)[1]
            if variable_name in state['variables']:
                return state['variables'][variable_name]
            else:
                raise ValueError(f'Variable {variable_name} not found.')

        if isinstance(argument, str) and argument.startswith('\'') and argument.endswith('\''):
            return argument[1:-1]

        try:
            return int(argument)
        except (ValueError, TypeError):
            return argument

    def _parse_and_execute_line(self, line, state):
        sub_op_match = re.search(r'\[(.*)\]', line)
        sub_op_string = None
        if sub_op_match:
            sub_op_string = sub_op_match.group(1).strip()
            line = line.replace(sub_op_match.group(0), '')

        parts = re.findall(r'\'[^\']*\'|var:[\w_]+|[\w_]+|-?\d+', line)
        operation_name = parts[0]
        if operation_name not in self.operations:
            raise ValueError(f'Unknown DSL operation: {operation_name}')

        parsed_args = [self._resolve_argument(arg, state) for arg in parts[1:]]

        if sub_op_string is not None:
            parsed_args.append(sub_op_string)

        return self.operations[operation_name](state, *parsed_args)

    def _find_objects(self, grid, background_color=0):
        height, width = grid.shape
        visited = torch.zeros_like(grid, dtype=torch.bool)
        objects = []

        for r in range(height):
            for c in range(width):
                color = grid[r, c].item()
                if not visited[r, c] and color != background_color:
                    obj = {
                        'color': color,
                        'pixels': set(),
                        'min_row': r, 'max_row': r,
                        'min_col': c, 'max_col': c,
                    }
                    queue = deque([(r, c)])
                    visited[r, c] = True
                    obj['pixels'].add((r, c))

                    while queue:
                        row, col = queue.popleft()
                        obj['min_row'] = min(obj['min_row'], row)
                        obj['max_row'] = max(obj['max_row'], row)
                        obj['min_col'] = min(obj['min_col'], col)
                        obj['max_col'] = max(obj['max_col'], col)

                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = row + dr, col + dc
                            if (0 <= nr < height and 0 <= nc < width and
                                        not visited[nr, nc] and
                                        grid[nr, nc].item() == color):
                                visited[nr, nc] = True
                                obj['pixels'].add((nr, nc))
                                queue.append((nr, nc))

                    obj['height'] = obj['max_row'] - obj['min_row'] + 1
                    obj['width'] = obj['max_col'] - obj['min_col'] + 1
                    obj['size'] = len(obj['pixels'])
                    objects.append(obj)
        return objects

    def _create_object_from_pixels(self, pixels, color):
        if not pixels:
            return None
        min_row = min(p[0] for p in pixels)
        max_row = max(p[0] for p in pixels)
        min_col = min(p[1] for p in pixels)
        max_col = max(p[1] for p in pixels)
        return {
            'color': color,
            'pixels': pixels,
            'min_row': min_row, 'max_row': max_row,
            'min_col': min_col, 'max_col': max_col,
            'height': max_row - min_row + 1,
            'width': max_col - min_col + 1,
            'size': len(pixels)
        }

    def _get_object_subgrid(self, obj):
        subgrid = torch.zeros((obj['height'], obj['width']), dtype=GRID_DTYPE)
        for r, c in obj['pixels']:
            subgrid[r - obj['min_row'], c - obj['min_col']] = obj['color']
        return subgrid

    def _get_center_of_object(self, obj):
        center_row = (obj['min_row'] + obj['max_row']) / 2
        center_col = (obj['min_col'] + obj['max_col']) / 2
        return center_row, center_col

    def _move_object(self, state, obj, dr, dc):
        for r, c in obj['pixels']:
            state['current_grid'][r, c] = 0

        new_pixels = set()
        for r, c in obj['pixels']:
            nr, nc = r + dr, c + dc
            if 0 <= nr < state['current_grid'].shape[0] and 0 <= nc < state['current_grid'].shape[1]:
                new_pixels.add((nr, nc))

        for r, c in new_pixels:
            state['current_grid'][r, c] = obj['color']

        return state

    def _are_objects_touching(self, obj1, obj2):
        for r1, c1 in obj1['pixels']:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (r1 + dr, c1 + dc) in obj2['pixels']:
                    return True
        return False

    def _get_obj_identifier(self, obj):
        return frozenset(obj['pixels'])

    def _find_all_objects_by_color_connectivity(self, state):
        state['all_objects'] = self._find_objects(state['current_grid'])
        return state

    def _select_all_found_objects(self, state):
        state['selected_objects'] = state['all_objects']
        return state

    def _clear_current_selection(self, state):
        state['selected_objects'] = []
        return state

    def _filter_selection_by_color(self, state, color_id):
        state['selected_objects'] = [
            obj for obj in state['selected_objects'] if obj['color'] == color_id
        ]
        return state

    def _filter_selection_by_exact_size(self, state, size):
        state['selected_objects'] = [
            obj for obj in state['selected_objects'] if obj['size'] == size
        ]
        return state

    def _filter_selection_by_size_greater_than(self, state, size):
        state['selected_objects'] = [
            obj for obj in state['selected_objects'] if obj['size'] > size
        ]
        return state

    def _filter_selection_by_size_less_than(self, state, size):
        state['selected_objects'] = [
            obj for obj in state['selected_objects'] if obj['size'] < size
        ]
        return state

    def _filter_selection_by_bounding_box(self, state, area):
        height, width = state['current_grid'].shape
        mid_row, mid_col = height // 2, width // 2
        filtered = []
        for obj in state['selected_objects']:
            if area == 'top_half' and obj['max_row'] < mid_row:
                filtered.append(obj)
            elif area == 'bottom_half' and obj['min_row'] >= mid_row:
                filtered.append(obj)
            elif area == 'left_half' and obj['max_col'] < mid_col:
                filtered.append(obj)
            elif area == 'right_half' and obj['min_col'] >= mid_col:
                filtered.append(obj)
        state['selected_objects'] = filtered
        return state

    def _filter_selection_by_touching_grid_border(self, state):
        height, width = state['current_grid'].shape
        filtered = []
        for obj in state['selected_objects']:
            if (obj['min_row'] == 0 or obj['max_row'] == height - 1 or
                    obj['min_col'] == 0 or obj['max_col'] == width - 1):
                filtered.append(obj)
        state['selected_objects'] = filtered
        return state

    def _filter_selection_by_not_touching_grid_border(self, state):
        height, width = state['current_grid'].shape
        filtered = []
        for obj in state['selected_objects']:
            if not (obj['min_row'] == 0 or obj['max_row'] == height - 1 or
                    obj['min_col'] == 0 or obj['max_col'] == width - 1):
                filtered.append(obj)
        state['selected_objects'] = filtered
        return state

    def _filter_selection_by_row_overlap(self, state, row_index):
        state['selected_objects'] = [
            obj for obj in state['selected_objects'] if obj['min_row'] <= row_index <= obj['max_row']
        ]
        return state

    def _filter_selection_by_column_overlap(self, state, col_index):
        state['selected_objects'] = [
            obj for obj in state['selected_objects'] if obj['min_col'] <= col_index <= obj['max_col']
        ]
        return state

    def _select_largest_objects_from_selection(self, state):
        if not state['selected_objects']: return state
        max_size = max(obj['size'] for obj in state['selected_objects'])
        state['selected_objects'] = [obj for obj in state['selected_objects'] if obj['size'] == max_size]
        return state

    def _select_smallest_objects_from_selection(self, state):
        if not state['selected_objects']: return state
        min_size = min(obj['size'] for obj in state['selected_objects'])
        state['selected_objects'] = [obj for obj in state['selected_objects'] if obj['size'] == min_size]
        return state

    def _select_object_from_selection_by_size_rank(self, state, rank, order='desc'):
        if not state['selected_objects']: return state
        sorted_objs = sorted(state['selected_objects'], key=lambda o: o['size'], reverse=(order == 'desc'))
        if rank < len(sorted_objs):
            state['selected_objects'] = [sorted_objs[rank]]
        else:
            state['selected_objects'] = []
        return state

    def _group_selection_by_color(self, state):
        groups = {}
        for obj in state['selected_objects']:
            color = obj['color']
            if color not in groups:
                groups[color] = []
            groups[color].append(obj)
        state['selection_groups'] = groups
        return state

    def _move_selected_objects_by_delta(self, state, dr, dc):
        for obj in state['selected_objects']:
            state = self._move_object(state, obj, dr, dc)
        return self._find_all_objects_by_color_connectivity(state)

    def _move_selected_objects_until_contact(self, state, direction):
        dr, dc = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (1, 0)}[direction]
        for steps in range(1, max(state['current_grid'].shape)):
            can_move = True
            for obj in state['selected_objects']:
                for r, c in obj['pixels']:
                    nr, nc = r + dr * steps, c + dc * steps
                    if not (0 <= nr < state['current_grid'].shape[0] and 0 <= nc < state['current_grid'].shape[1]):
                        can_move = False; break
                    pixel_is_occupied = state['current_grid'][nr, nc] != 0
                    pixel_is_part_of_selection = any((nr, nc) in o['pixels'] for o in state['selected_objects'])
                    if pixel_is_occupied and not pixel_is_part_of_selection:
                        can_move = False; break
                if not can_move: break
            if not can_move:
                return self._move_selected_objects_by_delta(state, dr * (steps - 1), dc * (steps - 1))
        return state

    def _extend_selected_objects_in_direction(self, state, direction, color_id):
        dr, dc = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (1, 0)}[direction]
        new_grid = state['current_grid'].clone()
        for obj in state['selected_objects']:
            for r_start, c_start in obj['pixels']:
                for i in range(1, max(new_grid.shape)):
                    r, c = r_start + dr * i, c_start + dc * i
                    if not (0 <= r < new_grid.shape[0] and 0 <= c < new_grid.shape[1]): break
                    if new_grid[r, c] != 0: break
                    new_grid[r, c] = color_id
        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _scale_selected_objects_from_top_left(self, state, factor):
        new_grid = state['current_grid'].clone()
        for obj in state['selected_objects']:
            for r, c in obj['pixels']: new_grid[r,c] = 0

            min_r, min_c = obj['min_row'], obj['min_col']
            new_pixels = set()
            for r, c in obj['pixels']:
                for i in range(factor):
                    for j in range(factor):
                        nr = min_r + (r - min_r) * factor + i
                        nc = min_c + (c - min_c) * factor + j
                        if 0 <= nr < new_grid.shape[0] and 0 <= nc < new_grid.shape[1]:
                            new_pixels.add((nr, nc))

            for r_new, c_new in new_pixels: new_grid[r_new, c_new] = obj['color']
        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _resize_grid_to_dimensions(self, state, height, width):
        new_grid = torch.zeros((height, width), dtype=GRID_DTYPE)
        old_h, old_w = state['current_grid'].shape
        h_copy = min(old_h, height)
        w_copy = min(old_w, width)
        new_grid[:h_copy, :w_copy] = state['current_grid'][:h_copy, :w_copy]
        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _rotate_selected_objects_around_center(self, state, angle):
        new_grid = state['current_grid'].clone()
        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                new_grid[r, c] = 0
            center_row, center_col = self._get_center_of_object(obj)
            new_pixels = set()
            for r, c in obj['pixels']:
                rel_row, rel_col = r - center_row, c - center_col
                if angle == 90:
                    new_rel_col, new_rel_row = -rel_row, rel_col
                elif angle == 180:
                    new_rel_col, new_rel_row = -rel_col, -rel_row
                elif angle == 270:
                    new_rel_col, new_rel_row = rel_row, -rel_col
                else:
                    continue
                new_r, new_c = round(center_row + new_rel_row), round(center_col + new_rel_col)
                new_pixels.add((new_r, new_c))

            for r_new, c_new in new_pixels:
                if 0 <= r_new < new_grid.shape[0] and 0 <= c_new < new_grid.shape[1]:
                    new_grid[int(r_new), int(c_new)] = obj['color']
        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _flip_selected_objects_around_center(self, state, axis):
        new_grid = state['current_grid'].clone()
        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                new_grid[r, c] = 0
            center_row, center_col = self._get_center_of_object(obj)
            new_pixels = set()
            for r, c in obj['pixels']:
                if axis == 'horizontal':
                    new_c = round(2 * center_col - c)
                    new_pixels.add((r, new_c))
                elif axis == 'vertical':
                    new_r = round(2 * center_row - r)
                    new_pixels.add((new_r, c))

            for r_new, c_new in new_pixels:
                if 0 <= r_new < new_grid.shape[0] and 0 <= c_new < new_grid.shape[1]:
                    new_grid[int(r_new), int(c_new)] = obj['color']
        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _recolor_selected_objects(self, state, new_color_id):
        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                state['current_grid'][r, c] = new_color_id
        return self._find_all_objects_by_color_connectivity(state)

    def _copy_selected_objects_in_place(self, state):
        new_objects_to_add = []
        for obj in state['selected_objects']:
            new_obj = obj.copy()
            new_objects_to_add.append(new_obj)
        state['all_objects'].extend(new_objects_to_add)
        state['selected_objects'].extend(new_objects_to_add)
        return state

    def _delete_selected_objects(self, state):
        return self._delete_pixels_of_selection(state)

    def _crop_grid_to_selection_bounding_box(self, state):
        if not state['selected_objects']: return state
        min_row = min(obj['min_row'] for obj in state['selected_objects'])
        max_row = max(obj['max_row'] for obj in state['selected_objects'])
        min_col = min(obj['min_col'] for obj in state['selected_objects'])
        max_col = max(obj['max_col'] for obj in state['selected_objects'])
        state['current_grid'] = state['current_grid'][min_row:max_row+1, min_col:max_col+1]
        return self._find_all_objects_by_color_connectivity(state)

    def _set_grid_background_color(self, state, color):
        state['current_grid'][state['current_grid'] == 0] = color
        return self._find_all_objects_by_color_connectivity(state)

    def _fill_entire_grid_with_color(self, state, color):
        state['current_grid'].fill_(color)
        return self._find_all_objects_by_color_connectivity(state)

    def _repeat_sub_operation(self, state, count, sub_operation_string):
        if isinstance(count, str):
            num_repeats = state['variables'].get(count, 0)
        else:
            num_repeats = count

        for _ in range(num_repeats):
            state = self._parse_and_execute_line(sub_operation_string, state)
        return state

    def _apply_sub_operation_to_each_selected_object(self, state, sub_operation_string):
        original_selection = list(state['selected_objects'])
        for obj in original_selection:
            state['selected_objects'] = [obj]
            state = self._parse_and_execute_line(sub_operation_string, state)
        state['selected_objects'] = original_selection
        return state

    def _apply_color_pattern_to_grid(self, state, pattern_str, direction):
        try:
            pattern = [int(c) for c in pattern_str.split(',')]
        except (ValueError, AttributeError):
            return state

        if not pattern:
            return state

        grid = state['current_grid']
        h, w = grid.shape
        pattern_len = len(pattern)

        if direction == 'row':
            for r in range(h):
                color = pattern[r % pattern_len]
                grid[r, :] = color
        elif direction == 'col' or direction == 'column':
            for c in range(w):
                color = pattern[c % pattern_len]
                grid[:, c] = color

        state['current_grid'] = grid
        return self._find_all_objects_by_color_connectivity(state)

    def _save_current_selection_to_memory(self, state, name):
        state['saved_selections'][name] = [obj.copy() for obj in state['selected_objects']]
        return state

    def _load_selection_from_memory(self, state, name):
        if name in state['saved_selections']:
            state['selected_objects'] = [obj.copy() for obj in state['saved_selections'][name]]
        return state

    def _clear_saved_selection_from_memory(self, state, name):
        if name in state['saved_selections']:
            del state['saved_selections'][name]
        return state

    def _count_selected_objects_to_variable(self, state, var_name):
        state['variables'][var_name] = len(state['selected_objects'])
        return state

    def _get_color_of_first_selected_object_to_variable(self, state, variable_name):
        if state['selected_objects']:
            state['variables'][variable_name] = state['selected_objects'][0]['color']
        return state

    def _get_size_of_first_selected_object_to_variable(self, state, variable_name):
        if state['selected_objects']:
            state['variables'][variable_name] = state['selected_objects'][0]['size']
        return state

    def _get_position_of_first_selected_object_to_variables(self, state, prefix):
        if state['selected_objects']:
            obj = state['selected_objects'][0]
            state['variables'][f'{prefix}_min_row'] = obj['min_row']
            state['variables'][f'{prefix}_max_row'] = obj['max_row']
            state['variables'][f'{prefix}_min_col'] = obj['min_col']
            state['variables'][f'{prefix}_max_col'] = obj['max_col']
        return state

    def _get_grid_dimensions_to_variables(self, state, height_var, width_var):
        height, width = state['current_grid'].shape
        state['variables'][height_var] = height
        state['variables'][width_var] = width
        return state

    def _set_variable_value(self, state, name, value):
        state['variables'][name] = value
        return state

    def _add_to_variable(self, state, name, value):
        if name in state['variables']:
            state['variables'][name] += value
        return state

    def _subtract_from_variable(self, state, name, value):
        if name in state['variables']:
            state['variables'][name] -= value
        return state

    def _multiply_variable_by(self, state, name, value):
        if name in state['variables']:
            state['variables'][name] *= value
        return state

    def _divide_variable_by(self, state, name, value):
        if name in state['variables'] and value != 0:
            state['variables'][name] //= value
        return state

    def _modulo_variable_by(self, state, name, value):
        if name in state['variables'] and value != 0:
            state['variables'][name] %= value
        return state

    def _intersect_current_selection_with_saved(self, state, name):
        if name not in state['saved_selections']:
            state['selected_objects'] = []
            return state
        saved_selection = state['saved_selections'][name]
        saved_ids = {self._get_obj_identifier(obj) for obj in saved_selection}
        state['selected_objects'] = [
            obj for obj in state['selected_objects'] if self._get_obj_identifier(obj) in saved_ids
        ]
        return state

    def _union_current_selection_with_saved(self, state, name):
        if name not in state['saved_selections']:
            return state
        saved_selection = state['saved_selections'][name]
        current_ids = {self._get_obj_identifier(obj) for obj in state['selected_objects']}
        for obj in saved_selection:
            if self._get_obj_identifier(obj) not in current_ids:
                state['selected_objects'].append(obj)
        return state

    def _xor_current_selection_with_saved(self, state, name):
        if name not in state['saved_selections']:
            return state
        saved_selection = state['saved_selections'][name]
        saved_ids = {self._get_obj_identifier(obj) for obj in saved_selection}
        current_ids = {self._get_obj_identifier(obj) for obj in state['selected_objects']}

        xor_ids = saved_ids.symmetric_difference(current_ids)

        all_objs = state['selected_objects'] + saved_selection
        unique_objs = {self._get_obj_identifier(obj): obj for obj in all_objs}

        state['selected_objects'] = [
            obj for obj_id, obj in unique_objs.items() if obj_id in xor_ids
        ]
        return state

    def _invert_current_selection(self, state):
        selected_ids = {self._get_obj_identifier(obj) for obj in state['selected_objects']}
        state['selected_objects'] = [
            obj for obj in state['all_objects'] if self._get_obj_identifier(obj) not in selected_ids
        ]
        return state

    def _group_selected_objects_by_connectivity(self, state):
        if len(state['selected_objects']) < 2:
            return state

        adj = {i: [] for i in range(len(state['selected_objects']))}
        for i in range(len(state['selected_objects'])):
            for j in range(i + 1, len(state['selected_objects'])):
                if self._are_objects_touching(state['selected_objects'][i], state['selected_objects'][j]):
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()
        new_objects = []
        for i in range(len(state['selected_objects'])):
            if i not in visited:
                component_indices = []
                q = deque([i])
                visited.add(i)
                while q:
                    u = q.popleft()
                    component_indices.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)

                all_pixels = set()
                base_color = state['selected_objects'][component_indices[0]]['color']
                for index in component_indices:
                    all_pixels.update(state['selected_objects'][index]['pixels'])

                new_obj = self._create_object_from_pixels(all_pixels, base_color)
                if new_obj:
                    new_objects.append(new_obj)

        state['selected_objects'] = new_objects
        return state

    def _get_center_of_selection_to_variables(self, state, x_var, y_var):
        if not state['selected_objects']:
            return state
        all_pixels = {p for obj in state['selected_objects'] for p in obj['pixels']}
        if not all_pixels:
            return state

        min_r = min(p[0] for p in all_pixels)
        max_r = max(p[0] for p in all_pixels)
        min_c = min(p[1] for p in all_pixels)
        max_c = max(p[1] for p in all_pixels)

        state['variables'][y_var] = (min_r + max_r) / 2
        state['variables'][x_var] = (min_c + max_c) / 2
        return state

    def _align_selected_objects(self, state, edge):
        if len(state['selected_objects']) < 2: return state

        if edge == 'top':
            target = min(o['min_row'] for o in state['selected_objects'])
            for obj in state['selected_objects']:
                dr = target - obj['min_row']
                state = self._move_object(state, obj, dr, 0)
        elif edge == 'left':
            target = min(o['min_col'] for o in state['selected_objects'])
            for obj in state['selected_objects']:
                dc = target - obj['min_col']
                state = self._move_object(state, obj, 0, dc)
        return self._find_all_objects_by_color_connectivity(state)

    def _distribute_selected_objects(self, state, direction, spacing):
        if len(state['selected_objects']) < 2:
            return state

        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                state['current_grid'][r, c] = 0

        if direction == 'horizontal':
            sorted_objs = sorted(state['selected_objects'], key=lambda o: o['min_col'])
            current_c = sorted_objs[0]['min_col']
            for obj in sorted_objs:
                dc = current_c - obj['min_col']
                self._move_object(state, obj, 0, dc)
                current_c += obj['width'] + spacing
        elif direction == 'vertical':
            sorted_objs = sorted(state['selected_objects'], key=lambda o: o['min_row'])
            current_r = sorted_objs[0]['min_row']
            for obj in sorted_objs:
                dr = current_r - obj['min_row']
                self._move_object(state, obj, dr, 0)
                current_r += obj['height'] + spacing

        return self._find_all_objects_by_color_connectivity(state)

    def _map_all_grid_colors(self, state, color_map_str):
        color_map = {int(k): int(v) for k, v in (pair.split(':') for pair in color_map_str.split(','))}
        new_grid = state['current_grid'].clone()
        for old_color, new_color in color_map.items():
            new_grid[state['current_grid'] == old_color] = new_color
        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _project_selection_onto_axis(self, state, axis):
        if not state['selected_objects']:
            return state

        all_pixels = {p for obj in state['selected_objects'] for p in obj['pixels']}
        if not all_pixels:
            return state

        min_r = min(p[0] for p in all_pixels)
        max_r = max(p[0] for p in all_pixels)
        min_c = min(p[1] for p in all_pixels)
        max_c = max(p[1] for p in all_pixels)

        if axis == 'horizontal':
            new_grid = torch.zeros((1, max_c - min_c + 1), dtype=GRID_DTYPE)
            for c in range(min_c, max_c + 1):
                count = sum(1 for r in range(min_r, max_r + 1) if (r, c) in all_pixels)
                if count > 0:
                    new_grid[0, c - min_c] = count % 10
        elif axis == 'vertical':
            new_grid = torch.zeros((max_r - min_r + 1, 1), dtype=GRID_DTYPE)
            for r in range(min_r, max_r + 1):
                count = sum(1 for c in range(min_c, max_c + 1) if (r, c) in all_pixels)
                if count > 0:
                    new_grid[r - min_r, 0] = count % 10
        else:
            return state

        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _copy_selection_and_apply_sub_operation(self, state, sub_operation_string):
        copied_objects = [o.copy() for o in state['selected_objects']]
        temp_state = {'selected_objects': copied_objects, 'current_grid': state['current_grid']}
        self._parse_and_execute_line(sub_operation_string, temp_state)
        return state

    def _mirror_selection_across_grid_axis(self, state, axis):
        new_grid = state['current_grid'].clone()
        grid_h, grid_w = new_grid.shape
        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                if axis == 'vertical':
                    nc = grid_w - 1 - c
                    if 0 <= nc < grid_w: new_grid[r, nc] = obj['color']
                elif axis == 'horizontal':
                    nr = grid_h - 1 - r
                    if 0 <= nr < grid_h: new_grid[nr, c] = obj['color']
        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _delete_pixels_of_selection(self, state):
        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                state['current_grid'][r, c] = 0
        return self._find_all_objects_by_color_connectivity(state)

    def _compose_selected_objects_into_new_grid(self, state, layout, padding=1):
        if not state['selected_objects']: return state

        if layout == 'horizontal':
            total_width = sum(o['width'] for o in state['selected_objects']) + padding * (len(state['selected_objects']) - 1)
            max_height = max(o['height'] for o in state['selected_objects'])
            new_grid = torch.zeros((max_height, total_width), dtype=GRID_DTYPE)

            current_c = 0
            for obj in state['selected_objects']:
                for r, c in obj['pixels']:
                    nr, nc = r - obj['min_row'], c - obj['min_col'] + current_c
                    new_grid[nr, nc] = obj['color']
                current_c += obj['width'] + padding
        elif layout == 'vertical':
            total_height = sum(o['height'] for o in state['selected_objects']) + padding * (len(state['selected_objects']) - 1)
            max_width = max(o['width'] for o in state['selected_objects'])
            new_grid = torch.zeros((total_height, max_width), dtype=GRID_DTYPE)

            current_r = 0
            for obj in state['selected_objects']:
                for r, c in obj['pixels']:
                    nr, nc = r - obj['min_row'] + current_r, c - obj['min_col']
                    new_grid[nr, nc] = obj['color']
                current_r += obj['height'] + padding

        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _get_property_of_first_selected_object_to_variable(self, state, property_name, variable_name):
        if state['selected_objects']:
            obj = state['selected_objects'][0]
            if property_name in obj:
                state['variables'][variable_name] = obj[property_name]
        return state

    def _execute_sub_operation_if_variable_equals(self, state, var_name, value, sub_operation_string):
        if state['variables'].get(var_name) == value:
            state = self._parse_and_execute_line(sub_operation_string, state)
        return state

    def _hollow_selected_objects(self, state):
        for obj in state['selected_objects']:
            interior_pixels = set()
            for r, c in obj['pixels']:
                is_interior = True
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in obj['pixels']:
                        is_interior = False; break
                if is_interior:
                    interior_pixels.add((r, c))
            for r, c in interior_pixels:
                state['current_grid'][r, c] = 0
        return self._find_all_objects_by_color_connectivity(state)

    def _rotate_selected_objects_90_degrees(self, state):
        return self._rotate_selected_objects_around_center(state, 90)

    def _rotate_selected_objects_180_degrees(self, state):
        return self._rotate_selected_objects_around_center(state, 180)

    def _rotate_selected_objects_270_degrees(self, state):
        return self._rotate_selected_objects_around_center(state, 270)

    def _filter_selection_by_symmetry(self, state, axis):
        symmetric_objects = []
        for obj in state['selected_objects']:
            obj_grid = self._get_object_subgrid(obj)
            is_symmetric = False
            if axis == 'horizontal':
                flipped = torch.flipud(obj_grid)
                is_symmetric = torch.equal(obj_grid, flipped)
            elif axis == 'vertical':
                flipped = torch.fliplr(obj_grid)
                is_symmetric = torch.equal(obj_grid, flipped)

            if is_symmetric:
                symmetric_objects.append(obj)
        state['selected_objects'] = symmetric_objects
        return state

    def _get_selection_bounding_box_to_variables(self, state, variable_name_prefix):
        if not state['selected_objects']:
            return state

        all_pixels = [p for obj in state['selected_objects'] for p in obj['pixels']]
        if not all_pixels:
            return state

        min_r = min(p[0] for p in all_pixels)
        max_r = max(p[0] for p in all_pixels)
        min_c = min(p[1] for p in all_pixels)
        max_c = max(p[1] for p in all_pixels)

        state['variables'][f'{variable_name_prefix}_min_row'] = min_r
        state['variables'][f'{variable_name_prefix}_max_row'] = max_r
        state['variables'][f'{variable_name_prefix}_min_col'] = min_c
        state['variables'][f'{variable_name_prefix}_max_col'] = max_c
        return state

    def _count_unique_colors_in_selection_to_variable(self, state, variable_name):
        if not state['selected_objects']:
            state['variables'][variable_name] = 0
            return state

        colors = {obj['color'] for obj in state['selected_objects']}
        state['variables'][variable_name] = len(colors)
        return state

    def _get_most_frequent_color_in_selection_to_variable(self, state, variable_name):
        if not state['selected_objects']:
            return state

        colors = [obj['color'] for obj in state['selected_objects']]
        if not colors:
            return state

        state['variables'][variable_name] = Counter(colors).most_common(1)[0][0]
        return state

    def _get_least_frequent_color_in_selection_to_variable(self, state, variable_name):
        if not state['selected_objects']:
            return state

        colors = [obj['color'] for obj in state['selected_objects']]
        if not colors:
            return state

        state['variables'][variable_name] = Counter(colors).most_common()[-1][0][0]
        return state

    def _flood_fill_from_selection(self, state, new_color):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            start_pixel = next(iter(obj['pixels']))
            original_color = state['current_grid'][start_pixel].item()

            q = deque([start_pixel])
            visited = {start_pixel}

            while q:
                r, c = q.popleft()
                state['current_grid'][r, c] = new_color

                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < state['current_grid'].shape[0] and
                        0 <= nc < state['current_grid'].shape[1] and
                        (nr, nc) not in visited and
                        state['current_grid'][nr, nc].item() == original_color):

                        visited.add((nr, nc))
                        q.append((nr, nc))

        return self._find_all_objects_by_color_connectivity(state)

    def _draw_line_between_centers_of_two_selected_objects(self, state, color):
        if len(state['selected_objects']) < 2:
            return state

        center1 = self._get_center_of_object(state['selected_objects'][0])
        center2 = self._get_center_of_object(state['selected_objects'][1])

        r1, c1 = int(center1[0]), int(center1[1])
        r2, c2 = int(center2[0]), int(center2[1])

        dr = abs(r2 - r1)
        dc = -abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr + dc

        while True:
            state['current_grid'][r1, c1] = color
            if r1 == r2 and c1 == c2:
                break
            e2 = 2 * err
            if e2 >= dc:
                err += dc
                r1 += sr
            if e2 <= dr:
                err += dr
                c1 += sc

        return self._find_all_objects_by_color_connectivity(state)

    def _draw_border_around_selection(self, state, color, thickness=1):
        if not state['selected_objects']:
            return state

        all_pixels_in_selection = {p for obj in state['selected_objects'] for p in obj['pixels']}
        border_pixels = set()

        for r_obj, c_obj in all_pixels_in_selection:
            for t in range(1, thickness + 1):
                for dr in range(-t, t + 1):
                    for dc in range(-t, t + 1):
                        if abs(dr) != t and abs(dc) != t:
                            continue

                        nr, nc = r_obj + dr, c_obj + dc
                        if (0 <= nr < state['current_grid'].shape[0] and
                            0 <= nc < state['current_grid'].shape[1] and
                            (nr, nc) not in all_pixels_in_selection):
                            border_pixels.add((nr, nc))

        for r, c in border_pixels:
            state['current_grid'][r, c] = color

        return self._find_all_objects_by_color_connectivity(state)

    def _invert_colors_in_selection(self, state, max_color=9):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                current_color = state['current_grid'][r, c].item()
                if current_color != 0:
                    state['current_grid'][r, c] = max_color - current_color + 1

        return self._find_all_objects_by_color_connectivity(state)

    def _shift_colors_in_selection(self, state, shift_amount=1):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                current_color = state['current_grid'][r, c].item()
                if current_color != 0:
                    new_color = (int(current_color) - 1 + shift_amount) % 9 + 1
                    state['current_grid'][r, c] = new_color

        return self._find_all_objects_by_color_connectivity(state)

    def _apply_kernel_from_variable_to_selection(self, state, kernel_variable_name):
        kernel_list = state['variables'].get(kernel_variable_name, [[0,0,0],[0,1,0],[0,0,0]])
        kernel = torch.tensor(kernel_list, dtype=GRID_DTYPE)
        if not state['selected_objects'] or kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            return state

        temp_grid = state['current_grid'].clone().unsqueeze(0).unsqueeze(0).float()
        padding = (kernel.shape[0] // 2, kernel.shape[1] // 2)

        output = torch.nn.functional.conv2d(temp_grid, kernel.unsqueeze(0).unsqueeze(0), padding=padding)

        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                state['current_grid'][r, c] = output[0, 0, r, c].round() % 10

        return self._find_all_objects_by_color_connectivity(state)

    def _filter_selection_by_neighbor_count(self, state, min_count, max_count):
        filtered_objects = []
        for i, obj1 in enumerate(state['selected_objects']):
            neighbor_count = 0
            for j, obj2 in enumerate(state['all_objects']):
                if obj1 is obj2: continue
                if self._are_objects_touching(obj1, obj2):
                    neighbor_count += 1
            if min_count <= neighbor_count <= max_count:
                filtered_objects.append(obj1)
        state['selected_objects'] = filtered_objects
        return state

    def _find_holes(self, obj, grid_shape):
        obj_grid = torch.zeros(grid_shape, dtype=torch.int)
        for r, c in obj['pixels']:
            obj_grid[r, c] = 1

        inverted_grid = 1 - obj_grid
        height, width = inverted_grid.shape
        visited = torch.zeros_like(inverted_grid, dtype=torch.bool)
        all_components = []

        for r in range(height):
            for c in range(width):
                if inverted_grid[r, c] == 1 and not visited[r, c]:
                    component = []
                    q = deque([(r, c)])
                    visited[r, c] = True
                    is_hole = True

                    while q:
                        curr_r, curr_c = q.popleft()
                        component.append((curr_r, curr_c))

                        if curr_r == 0 or curr_r == height - 1 or curr_c == 0 or curr_c == width - 1:
                            is_hole = False

                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if (0 <= nr < height and 0 <= nc < width and
                                not visited[nr, nc] and inverted_grid[nr, nc] == 1):
                                visited[nr, nc] = True
                                q.append((nr, nc))

                    if is_hole:
                        all_components.append(component)
        return all_components

    def _select_holes_within_selection(self, state):
        if not state['selected_objects']:
            return state

        hole_objects = []
        for i, obj in enumerate(state['selected_objects']):
            holes = self._find_holes(obj, state['current_grid'].shape)
            for j, hole_pixels in enumerate(holes):
                hole_obj = self._create_object_from_pixels(set(hole_pixels), color=i * 100 + j + 1)
                if hole_obj:
                    hole_objects.append(hole_obj)

        state['selected_objects'] = hole_objects
        return state

    def _fill_holes_within_selection(self, state):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            holes = self._find_holes(obj, state['current_grid'].shape)
            fill_color = obj['color']
            for hole_pixels in holes:
                for r, c in hole_pixels:
                    state['current_grid'][r, c] = fill_color

        return self._find_all_objects_by_color_connectivity(state)

    def _filter_selection_by_aspect_ratio(self, state, min_ratio, max_ratio):
        filtered = []
        for obj in state['selected_objects']:
            height = obj.get('height', 0)
            width = obj.get('width', 0)
            if height > 0 and width > 0:
                aspect_ratio = width / height
                if min_ratio <= aspect_ratio <= max_ratio:
                    filtered.append(obj)
        state['selected_objects'] = filtered
        return state

    def _stack_selected_objects_vertically(self, state, padding=0):
        if len(state['selected_objects']) < 2:
            return state

        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                state['current_grid'][r, c] = 0

        sorted_objs = sorted(state['selected_objects'], key=lambda o: o['min_row'])

        current_r = sorted_objs[0]['min_row']
        anchor_c = sorted_objs[0]['min_col']

        for obj in sorted_objs:
            for r_obj, c_obj in obj['pixels']:
                new_r = current_r + (r_obj - obj['min_row'])
                new_c = anchor_c + (c_obj - obj['min_col'])
                if 0 <= new_r < state['current_grid'].shape[0] and 0 <= new_c < state['current_grid'].shape[1]:
                    state['current_grid'][new_r, new_c] = obj['color']
            current_r += obj['height'] + padding

        return self._find_all_objects_by_color_connectivity(state)

    def _stack_selected_objects_horizontally(self, state, padding=0):
        if len(state['selected_objects']) < 2:
            return state

        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                state['current_grid'][r, c] = 0

        sorted_objs = sorted(state['selected_objects'], key=lambda o: o['min_col'])

        current_c = sorted_objs[0]['min_col']
        anchor_r = sorted_objs[0]['min_row']

        for obj in sorted_objs:
            for r_obj, c_obj in obj['pixels']:
                new_r = anchor_r + (r_obj - obj['min_row'])
                new_c = current_c + (c_obj - obj['min_col'])
                if 0 <= new_r < state['current_grid'].shape[0] and 0 <= new_c < state['current_grid'].shape[1]:
                    state['current_grid'][new_r, new_c] = obj['color']
            current_c += obj['width'] + padding

        return self._find_all_objects_by_color_connectivity(state)

    def _wrap_grid_by_delta(self, state, dr, dc):
        state['current_grid'] = torch.roll(state['current_grid'], shifts=(dr, dc), dims=(0, 1))
        return self._find_all_objects_by_color_connectivity(state)

    def _get_pixel_color_to_variable(self, state, r, c, variable_name):
        if 0 <= r < state['current_grid'].shape[0] and 0 <= c < state['current_grid'].shape[1]:
            state['variables'][variable_name] = state['current_grid'][r, c].item()
        return state

    def _set_pixel_color(self, state, r, c, color):
        if 0 <= r < state['current_grid'].shape[0] and 0 <= c < state['current_grid'].shape[1]:
            state['current_grid'][r, c] = color
        return self._find_all_objects_by_color_connectivity(state)

    def _is_object_convex(self, obj):
        if not obj['pixels'] or len(obj['pixels']) < 3:
            return True

        points = np.array(list(obj['pixels']))
        try:
            hull = ConvexHull(points)
            return len(hull.vertices) == len(points)
        except Exception:
            return True

    def _filter_selection_by_convexity(self, state):
        convex_objects = []
        for obj in state['selected_objects']:
            if self._is_object_convex(obj):
                convex_objects.append(obj)
        state['selected_objects'] = convex_objects
        return state

    def _transform_selection_to_convex_hull(self, state):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            points = np.array(list(obj['pixels']))
            if len(points) < 3:
                continue

            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            min_r, min_c = np.min(hull_points, axis=0)
            max_r, max_c = np.max(hull_points, axis=0)

            grid_r, grid_c = np.mgrid[min_r:max_r+1, min_c:max_c+1]
            coords = np.vstack((grid_r.ravel(), grid_c.ravel())).T

            delaunay = Delaunay(hull_points)
            in_hull = delaunay.find_simplex(coords) >= 0

            hull_pixels = set(tuple(p) for p in coords[in_hull])

            for r, c in obj['pixels']: state['current_grid'][r, c] = 0
            for r, c in hull_pixels: state['current_grid'][int(r), int(c)] = obj['color']

        return self._find_all_objects_by_color_connectivity(state)

    def _mask_selection_with_saved_selection(self, state, saved_mask_name):
        if not state['selected_objects'] or saved_mask_name not in state['saved_selections']:
            return state

        mask_objects = state['saved_selections'][saved_mask_name]
        mask_pixels = {p for obj in mask_objects for p in obj['pixels']}

        for obj in state['selected_objects']:
            pixels_to_erase = {p for p in obj['pixels'] if p not in mask_pixels}
            for r, c in pixels_to_erase:
                state['current_grid'][r, c] = 0

        return self._find_all_objects_by_color_connectivity(state)

    def _count_islands_of_color_to_variable(self, state, variable_name, color):
        grid_of_color = (state['current_grid'] == color).int()
        num_islands = 0
        visited = torch.zeros_like(grid_of_color, dtype=torch.bool)

        for r in range(grid_of_color.shape[0]):
            for c in range(grid_of_color.shape[1]):
                if grid_of_color[r, c] == 1 and not visited[r, c]:
                    num_islands += 1
                    q = deque([(r, c)])
                    visited[r, c] = True
                    while q:
                        curr_r, curr_c = q.popleft()
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if (0 <= nr < grid_of_color.shape[0] and
                                0 <= nc < grid_of_color.shape[1] and
                                not visited[nr, nc] and grid_of_color[nr, nc] == 1):
                                visited[nr, nc] = True
                                q.append((nr, nc))

        state['variables'][variable_name] = num_islands
        return state

    def _sort_selection_by_position(self, state):
        state['selected_objects'] = sorted(state['selected_objects'], key=lambda o: (o['min_row'], o['min_col']))
        return state

    def _shear_selection(self, state, shear_factor, axis):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            new_pixels = set()
            min_coord = obj['min_row'] if axis == 'horizontal' else obj['min_col']

            for r, c in obj['pixels']: state['current_grid'][r, c] = 0

            for r, c in obj['pixels']:
                new_r, new_c = r, c
                if axis == 'horizontal':
                    offset = int((r - min_coord) * shear_factor)
                    new_c = c + offset
                else:
                    offset = int((c - min_coord) * shear_factor)
                    new_r = r + offset

                if (0 <= new_r < state['current_grid'].shape[0] and 0 <= new_c < state['current_grid'].shape[1]):
                    new_pixels.add((new_r, new_c))

            for r_new, c_new in new_pixels:
                state['current_grid'][r_new, c_new] = obj['color']

        return self._find_all_objects_by_color_connectivity(state)

    def _split_grid_at_position_to_memory(self, state, axis, position, saved_name_1, saved_name_2):
        h, w = state['current_grid'].shape
        if axis == 'horizontal':
            if 0 < position < h:
                state['saved_selections'][saved_name_1] = state['current_grid'][:position, :].clone()
                state['saved_selections'][saved_name_2] = state['current_grid'][position:, :].clone()
        elif axis == 'vertical':
            if 0 < position < w:
                state['saved_selections'][saved_name_1] = state['current_grid'][:, :position].clone()
                state['saved_selections'][saved_name_2] = state['current_grid'][:, position:].clone()
        return state

    def _merge_grids_from_memory(self, state, saved_name_1, saved_name_2, axis):
        if saved_name_1 not in state['saved_selections'] or saved_name_2 not in state['saved_selections']:
            return state

        grid1 = state['saved_selections'][saved_name_1]
        grid2 = state['saved_selections'][saved_name_2]

        if axis == 'horizontal':
            new_grid = torch.cat((grid1, grid2), dim=0)
        elif axis == 'vertical':
            new_grid = torch.cat((grid1, grid2), dim=1)
        else:
            return state

        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)

    def _filter_selection_by_top_left_parity(self, state, row_parity, col_parity):
        filtered_objects = []
        for obj in state['selected_objects']:
            r, c = obj['min_row'], obj['min_col']
            row_match = (row_parity == 'even' and r % 2 == 0) or (row_parity == 'odd' and r % 2 != 0)
            col_match = (col_parity == 'even' and c % 2 == 0) or (col_parity == 'odd' and c % 2 != 0)
            if row_match and col_match:
                filtered_objects.append(obj)
        state['selected_objects'] = filtered_objects
        return state

    def _keep_only_diagonal_pixels_of_selection(self, state, direction):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            obj_grid = self._get_object_subgrid(obj)
            h, w = obj_grid.shape

            pixels_to_keep = set()
            for r in range(h):
                for c in range(w):
                    is_on_diag = (direction == 'main' and r == c) or (direction == 'anti' and r + c == w -1)
                    if is_on_diag:
                        pixels_to_keep.add((obj['min_row'] + r, obj['min_col'] + c))

            pixels_to_erase = obj['pixels'] - pixels_to_keep
            for r_erase, c_erase in pixels_to_erase:
                state['current_grid'][r_erase, c_erase] = 0

        return self._find_all_objects_by_color_connectivity(state)

    def _move_selection_to_grid_border(self, state, border):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            dr, dc = 0, 0
            if border == 'top':
                dr = -obj['min_row']
            elif border == 'bottom':
                dr = state['current_grid'].shape[0] - 1 - obj['max_row']
            elif border == 'left':
                dc = -obj['min_col']
            elif border == 'right':
                dc = state['current_grid'].shape[1] - 1 - obj['max_col']

            if dr != 0 or dc != 0:
                state = self._move_object(state, obj, dr, dc)

        return self._find_all_objects_by_color_connectivity(state)

    def _swap_positions_of_two_selected_objects(self, state):
        if len(state['selected_objects']) != 2:
            return state

        obj1, obj2 = state['selected_objects'][0], state['selected_objects'][1]

        center1 = self._get_center_of_object(obj1)
        center2 = self._get_center_of_object(obj2)
        dr, dc = int(center2[0] - center1[0]), int(center2[1] - center1[1])

        for r, c in obj1['pixels']: state['current_grid'][r, c] = 0
        for r, c in obj2['pixels']: state['current_grid'][r, c] = 0

        for r, c in obj1['pixels']:
            nr, nc = r + dr, c + dc
            if 0 <= nr < state['current_grid'].shape[0] and 0 <= nc < state['current_grid'].shape[1]:
                state['current_grid'][nr, nc] = obj1['color']

        for r, c in obj2['pixels']:
            nr, nc = r - dr, c - dc
            if 0 <= nr < state['current_grid'].shape[0] and 0 <= nc < state['current_grid'].shape[1]:
                state['current_grid'][nr, nc] = obj2['color']

        return self._find_all_objects_by_color_connectivity(state)

    def _crop_grid_to_all_content(self, state, padding=0):
        non_bg_pixels = (state['current_grid'] != 0).nonzero()
        if non_bg_pixels.numel() == 0:
            return state

        min_r = non_bg_pixels[:, 0].min().item()
        max_r = non_bg_pixels[:, 0].max().item()
        min_c = non_bg_pixels[:, 1].min().item()
        max_c = non_bg_pixels[:, 1].max().item()

        min_r = max(0, min_r - padding)
        max_r = min(state['current_grid'].shape[0] - 1, max_r + padding)
        min_c = max(0, min_c - padding)
        max_c = min(state['current_grid'].shape[1] - 1, max_c + padding)

        state['current_grid'] = state['current_grid'][min_r:max_r+1, min_c:max_c+1]
        return self._find_all_objects_by_color_connectivity(state)

    def _pad_grid_with_color(self, state, padding, color=0):
        state['current_grid'] = torch.nn.functional.pad(state['current_grid'], (padding, padding, padding, padding), mode='constant', value=color)
        return self._find_all_objects_by_color_connectivity(state)

    def _count_all_colors_in_grid_to_variables(self, state, variable_name_prefix):
        colors, counts = torch.unique(state['current_grid'], return_counts=True)
        for color, count in zip(colors, counts):
            state['variables'][f'{variable_name_prefix}_{int(color.item())}'] = count.item()
        return state

    def _select_objects_contained_within_others(self, state):
        contained_objects = []
        for i, obj1 in enumerate(state['all_objects']):
            for j, obj2 in enumerate(state['all_objects']):
                if i == j: continue
                if (obj2['min_row'] > obj1['min_row'] and
                    obj2['max_row'] < obj1['max_row'] and
                    obj2['min_col'] > obj1['min_col'] and
                    obj2['max_col'] < obj1['max_col']):

                    is_truly_inside = all(p not in obj1['pixels'] for p in obj2['pixels'])
                    if is_truly_inside:
                        contained_objects.append(obj2)
        state['selected_objects'] = contained_objects
        return state

    def _move_selection_until_collision(self, state, dr, dc):
        if not state['selected_objects']:
            return state

        static_objects = [obj for obj in state['all_objects'] if obj not in state['selected_objects']]

        for obj_to_move in state['selected_objects']:
            current_dr, current_dc = 0, 0
            while True:
                next_dr, next_dc = current_dr + dr, current_dc + dc
                collided = False

                moved_pixels = {(r + next_dr, c + next_dc) for r in obj_to_move['pixels']}
                for r_moved, c_moved in moved_pixels:
                    if not (0 <= r_moved < state['current_grid'].shape[0] and 0 <= c_moved < state['current_grid'].shape[1]):
                        collided = True; break
                if collided: break

                for static_obj in static_objects:
                    if not moved_pixels.isdisjoint(static_obj['pixels']):
                        collided = True; break
                if collided: break

                current_dr, current_dc = next_dr, next_dc

            if current_dr != 0 or current_dc != 0:
                state = self._move_object(state, obj_to_move, current_dr, current_dc)

        return self._find_all_objects_by_color_connectivity(state)

    def _get_grid_color_palette_to_variable(self, state, variable_name):
        colors = torch.unique(state['current_grid']).int().tolist()
        state['variables'][variable_name] = colors
        return state

    def _replace_color_within_selection(self, state, old_color, new_color):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            if obj['color'] == old_color:
                for r, c in obj['pixels']:
                    state['current_grid'][r, c] = new_color

        return self._find_all_objects_by_color_connectivity(state)

    def _create_object_from_selection_bounding_box(self, state, color):
        if not state['selected_objects']:
            return state

        min_r = min(o['min_row'] for o in state['selected_objects'])
        max_r = max(o['max_row'] for o in state['selected_objects'])
        min_c = min(o['min_col'] for o in state['selected_objects'])
        max_c = max(o['max_col'] for o in state['selected_objects'])

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                state['current_grid'][r, c] = color

        return self._find_all_objects_by_color_connectivity(state)

    def _select_background_as_object(self, state):
        h, w = state['current_grid'].shape
        bg_pixels = set()
        for r in range(h):
            for c in range(w):
                if state['current_grid'][r, c] == 0:
                    bg_pixels.add((r, c))

        if bg_pixels:
            bg_object = self._create_object_from_pixels(bg_pixels, 0)
            state['selected_objects'] = [bg_object]
        else:
            state['selected_objects'] = []
        return state

    def _filter_selection_by_hole_count(self, state, count):
        filtered = []
        for obj in state['selected_objects']:
            holes = self._find_holes(obj, state['current_grid'].shape)
            if len(holes) == count:
                filtered.append(obj)
        state['selected_objects'] = filtered
        return state

    def _get_object_hole_count_to_variable(self, state, variable_name):
        if not state['selected_objects']:
            return state
        holes = self._find_holes(state['selected_objects'][0], state['current_grid'].shape)
        state['variables'][variable_name] = len(holes)
        return state

    def _union_all_selected_objects(self, state):
        if not state['selected_objects']:
            return state

        all_pixels = {p for obj in state['selected_objects'] for p in obj['pixels']}
        color = state['selected_objects'][0]['color']

        new_obj = self._create_object_from_pixels(all_pixels, color)
        if new_obj:
            state['selected_objects'] = [new_obj]
        else:
            state['selected_objects'] = []
        return state

    def _recolor_object_border(self, state, color):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                is_border = False
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if (r + dr, c + dc) not in obj['pixels']:
                        is_border = True
                        break
                if is_border:
                    state['current_grid'][r, c] = color
        return self._find_all_objects_by_color_connectivity(state)

    def _recolor_object_interior(self, state, color):
        if not state['selected_objects']:
            return state

        for obj in state['selected_objects']:
            for r, c in obj['pixels']:
                is_interior = True
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if (r + dr, c + dc) not in obj['pixels']:
                        is_interior = False
                        break
                if is_interior:
                    state['current_grid'][r, c] = color
        return self._find_all_objects_by_color_connectivity(state)

    def _smear_selection_in_direction(self, state, direction, steps):
        dr, dc = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (1, 0)}[direction]
        for obj in state['selected_objects']:
            for r, c in list(obj['pixels']):
                for i in range(1, steps + 1):
                    nr, nc = r + dr * i, c + dc * i
                    if 0 <= nr < state['current_grid'].shape[0] and 0 <= nc < state['current_grid'].shape[1]:
                        state['current_grid'][nr, nc] = obj['color']
        return self._find_all_objects_by_color_connectivity(state)

    def _create_checkerboard_pattern(self, state, color1, color2, size):
        h, w = state['current_grid'].shape
        for r in range(h):
            for c in range(w):
                if ((r // size) + (c // size)) % 2 == 0:
                    state['current_grid'][r, c] = color1
                else:
                    state['current_grid'][r, c] = color2
        return self._find_all_objects_by_color_connectivity(state)

    def _tile_grid_with_selection(self, state):
        if not state['selected_objects']: return state
        obj_to_tile = state['selected_objects'][0]
        obj_h, obj_w = obj_to_tile['height'], obj_to_tile['width']
        if obj_h == 0 or obj_w == 0: return state
        grid_h, grid_w = state['current_grid'].shape
        for r in range(0, grid_h, obj_h):
            for c in range(0, grid_w, obj_w):
                for pr, pc in obj_to_tile['pixels']:
                    nr, nc = r + (pr - obj_to_tile['min_row']), c + (pc - obj_to_tile['min_col'])
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        state['current_grid'][nr, nc] = obj_to_tile['color']
        return self._find_all_objects_by_color_connectivity(state)

    def _get_object_perimeter_to_variable(self, state, variable_name):
        if not state['selected_objects']: return state
        obj = state['selected_objects'][0]
        perimeter = 0
        for r, c in obj['pixels']:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (r + dr, c + dc) not in obj['pixels']:
                    perimeter += 1
        state['variables'][variable_name] = perimeter
        return state

    def _filter_selection_by_perimeter_length(self, state, length):
        filtered = []
        for obj in state['selected_objects']:
            perimeter = 0
            for r, c in obj['pixels']:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if (r + dr, c + dc) not in obj['pixels']:
                        perimeter += 1
            if perimeter == length:
                filtered.append(obj)
        state['selected_objects'] = filtered
        return state

    def _select_objects_in_row(self, state, row_index):
        state['selected_objects'] = [
            obj for obj in state['all_objects'] if obj['min_row'] <= row_index <= obj['max_row']
        ]
        return state

    def _select_objects_in_column(self, state, col_index):
        state['selected_objects'] = [
            obj for obj in state['all_objects'] if obj['min_col'] <= col_index <= obj['max_col']
        ]
        return state

    def _get_object_shape_as_boolean_mask_to_variable(self, state, variable_name):
        if not state['selected_objects']: return state
        obj = state['selected_objects'][0]
        mask = torch.zeros((obj['height'], obj['width']), dtype=torch.bool)
        for r, c in obj['pixels']:
            mask[r - obj['min_row'], c - obj['min_col']] = True
        state['variables'][variable_name] = mask
        return state

    def _create_line_object(self, state, r1, c1, r2, c2, color):
        dr = abs(r2 - r1)
        dc = -abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr + dc
        while True:
            state['current_grid'][r1, c1] = color
            if r1 == r2 and c1 == c2: break
            e2 = 2 * err
            if e2 >= dc: err += dc; r1 += sr
            if e2 <= dr: err += dr; c1 += sc
        return self._find_all_objects_by_color_connectivity(state)

    def _move_object_to_absolute_position(self, state, r, c):
        if not state['selected_objects']: return state
        obj = state['selected_objects'][0]
        dr, dc = r - obj['min_row'], c - obj['min_col']
        return self._move_object(state, obj, dr, dc)

    def _select_object_at_absolute_position(self, state, r, c):
        state['selected_objects'] = [
            obj for obj in state['all_objects'] if (r, c) in obj['pixels']
        ]
        return state

    def _extrude_selection(self, state, dr, dc, steps):
        for _ in range(steps):
            for obj in state['selected_objects']:
                new_pixels = set()
                for r, c in obj['pixels']:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < state['current_grid'].shape[0] and 0 <= nc < state['current_grid'].shape[1]:
                        state['current_grid'][nr, nc] = obj['color']
                        new_pixels.add((nr, nc))
                obj['pixels'].update(new_pixels)
        return self._find_all_objects_by_color_connectivity(state)

    def _get_pixel_neighbor_colors_to_variable(self, state, r, c, variable_name):
        neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < state['current_grid'].shape[0] and 0 <= nc < state['current_grid'].shape[1]:
                neighbors.append(state['current_grid'][nr, nc].item())
        state['variables'][variable_name] = neighbors
        return state

    def _count_objects_in_row_to_variable(self, state, row_index, variable_name):
        count = sum(1 for obj in state['all_objects'] if obj['min_row'] <= row_index <= obj['max_row'])
        state['variables'][variable_name] = count
        return state

    def _count_objects_in_column_to_variable(self, state, col_index, variable_name):
        count = sum(1 for obj in state['all_objects'] if obj['min_col'] <= col_index <= obj['max_col'])
        state['variables'][variable_name] = count
        return state

    def _split_selection_by_color(self, state):
        if not state['selected_objects']: return state
        new_selection = []
        for obj in state['selected_objects']:
            subgrid = self._get_object_subgrid(obj)
            sub_objects = self._find_objects(subgrid)
            for sub_obj in sub_objects:
                pixels_in_main_grid = {(r + obj['min_row'], c + obj['min_col']) for r, c in sub_obj['pixels']}
                new_obj = self._create_object_from_pixels(pixels_in_main_grid, sub_obj['color'])
                if new_obj:
                    new_selection.append(new_obj)
        state['selected_objects'] = new_selection
        return state

    def _create_grid_from_selection(self, state, variable_name):
        if not state['selected_objects']: return state
        obj = state['selected_objects'][0]
        subgrid = self._get_object_subgrid(obj)
        state['variables'][variable_name] = subgrid
        return state

    def _overlay_grid_from_variable(self, state, variable_name):
        if variable_name not in state['variables']: return state
        grid_to_overlay = state['variables'][variable_name]
        h = min(state['current_grid'].shape[0], grid_to_overlay.shape[0])
        w = min(state['current_grid'].shape[1], grid_to_overlay.shape[1])
        for r in range(h):
            for c in range(w):
                if grid_to_overlay[r, c] != 0:
                    state['current_grid'][r, c] = grid_to_overlay[r, c]
        return self._find_all_objects_by_color_connectivity(state)

    def _select_objects_by_shape_template(self, state, template_variable):
        if template_variable not in state['variables']: return state
        template = state['variables'][template_variable]
        th, tw = template.shape
        filtered = []
        for obj in state['selected_objects']:
            if obj['height'] == th and obj['width'] == tw:
                obj_grid = self._get_object_subgrid(obj)
                obj_mask = (obj_grid > 0)
                if torch.equal(obj_mask, template):
                    filtered.append(obj)
        state['selected_objects'] = filtered
        return state

    def _apply_cellular_automaton_rule_to_selection(self, state, color):
        if not state['selected_objects']: return state
        grid = state['current_grid']
        new_grid = grid.clone()
        for obj in state['selected_objects']:
            for r in range(obj['min_row'], obj['max_row'] + 1):
                for c in range(obj['min_col'], obj['max_col'] + 1):
                    live_neighbors = 0
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == color:
                                live_neighbors += 1

                    is_alive = (grid[r, c] == color)
                    if is_alive and (live_neighbors < 2 or live_neighbors > 3):
                        new_grid[r, c] = 0
                    elif not is_alive and live_neighbors == 3:
                        new_grid[r, c] = color
        state['current_grid'] = new_grid
        return self._find_all_objects_by_color_connectivity(state)