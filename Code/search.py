import torch
from collections import namedtuple

State = namedtuple('State', ['partial_script', 'current_grids'])


class SolutionLogger:
    def __init__(self):
        self.solutions = []
        self.solution_hashes = set()

    def add_solution(self, script):
        script_tuple = tuple(script)
        if script_tuple not in self.solution_hashes:
            self.solutions.append({'script': script, 'length': len(script)})
            self.solution_hashes.add(script_tuple)

    def get_best_solution(self):
        if not self.solutions:
            return None

        sorted_solutions = sorted(self.solutions, key=lambda x: x['length'])
        best_script = sorted_solutions[0]['script']
        return '\n'.join(best_script)


class NGPSSearch:
    def __init__(self, llm_guide, dsl_executor, utils_module, beam_width=128, max_depth=32):
        self.llm_guide = llm_guide
        self.dsl_executor = dsl_executor
        self.utils = utils_module
        self.beam_width = beam_width
        self.max_depth = max_depth

    def find_program_for_task(self, base_prompt, training_pairs):
        solution_logger = SolutionLogger()

        initial_grids = [torch.tensor(pair['input'], dtype=torch.bfloat16) for pair in training_pairs]
        target_grids = [torch.tensor(pair['output'], dtype=torch.bfloat16) for pair in training_pairs]

        initial_state = State(partial_script=[], current_grids=initial_grids)

        beam = [initial_state]
        visited_grids_hash = {self._grids_to_hash(initial_grids)}

        for depth in range(self.max_depth):
            if not beam:
                print(f'Search ended at depth {depth}: Beam is empty.')
                break

            prompts = [self._build_step_prompt(base_prompt, training_pairs, state) for state in beam]
            generated_responses = self.llm_guide.get_next_command(prompts, num_candidates=4)

            all_candidate_states = []
            for i, state in enumerate(beam):
                next_command_candidates = generated_responses[i]

                for next_command in next_command_candidates:
                    if not next_command:
                        continue

                    full_script_so_far = state.partial_script + [next_command]
                    script_string = '\n'.join(full_script_so_far)

                    output_grids = [self.dsl_executor.execute_script(script_string, grid) for grid in initial_grids]

                    if any(grid is None for grid in output_grids):
                        continue

                    grids_hash = self._grids_to_hash(output_grids)
                    if grids_hash not in visited_grids_hash:
                        visited_grids_hash.add(grids_hash)
                        new_state = State(partial_script=full_script_so_far, current_grids=output_grids)
                        all_candidate_states.append(new_state)

            valid_next_states = []
            for candidate_state in all_candidate_states:
                if self._verify_state(candidate_state, target_grids):
                    solution_logger.add_solution(candidate_state.partial_script)
                valid_next_states.append(candidate_state)

            if not valid_next_states:
                print(f'Search ended at depth {depth}: No valid next states found.')
                break

            beam = sorted(valid_next_states, key=lambda s: len(s.partial_script))[:self.beam_width]

        return solution_logger.get_best_solution()

    def _get_grid_analysis_string(self, state, initial_grid_for_state):
        analysis_parts = []
        grid = state.current_grids[0]

        height, width = grid.shape
        analysis_parts.append(f'- Grid Dimensions: {height}x{width}')

        unique_colors = torch.unique(grid).int().tolist()
        analysis_parts.append(f'- Color Palette: {unique_colors}')

        script_so_far = '\n'.join(state.partial_script)
        temp_dsl_state = self.dsl_executor.execute_script(script_so_far, initial_grid_for_state)

        if temp_dsl_state is None:
            analysis_parts.append('- Analysis failed: script execution resulted in an error.')
            return '\n'.join(analysis_parts)

        all_objects = temp_dsl_state.get('all_objects', [])
        selected_objects = temp_dsl_state.get('selected_objects', [])

        if not all_objects:
            analysis_parts.append('- No objects found on the grid.')
        else:
            analysis_parts.append(f'- Total Objects Found: {len(all_objects)}')

            if selected_objects:
                analysis_parts.append(f'- Currently Selected Objects ({len(selected_objects)}):')
                for i, obj in enumerate(selected_objects[:3]):
                    pos = f'({obj["min_row"]},{obj["min_col"]}) to ({obj["max_row"]},{obj["max_col"]})'
                    analysis_parts.append(f'  - Obj {i+1}: color={obj["color"]}, size={obj["size"]}, pos={pos}')
                if len(selected_objects) > 3:
                    analysis_parts.append('  - ... and more.')
            else:
                analysis_parts.append('- No objects are currently selected.')

        return '\n'.join(analysis_parts)

    def _build_step_prompt(self, base_prompt, training_pairs, state):
        prompt_parts = [base_prompt]

        representative_pair = training_pairs[0]
        representative_current_grid = state.current_grids[0]
        initial_grid_for_analysis = torch.tensor(representative_pair['input'], dtype=torch.bfloat16)

        input_grid_str = self.utils.format_grid_to_string(representative_pair['input'])
        output_grid_str = self.utils.format_grid_to_string(representative_pair['output'])
        example_str = (
            '### PUZZLE EXAMPLE (1 of ' + str(len(training_pairs)) + ') ###\n'
            f'INPUT_GRID:\n\'{input_grid_str}\'\n\n'
            f'GOAL_OUTPUT_GRID:\n\'{output_grid_str}\''
        )
        prompt_parts.append(example_str)

        state_header = '### CURRENT STATE (of Example 1) ###'
        if state.partial_script:
            script_str = '\n'.join(state.partial_script)
            current_grid_str = self.utils.format_grid_to_string(representative_current_grid)
            state_description = (
                f'The following partial script has been applied:\n{script_str}\n\n'
                f'This has transformed the input into the following current grid:\n\'{current_grid_str}\''
            )
            prompt_parts.extend([state_header, state_description])
        else:
            state_description = 'No operations have been applied yet. The current grid is the original input grid.'
            prompt_parts.extend([state_header, state_description])

        analysis_header = '### GRID ANALYSIS (of Example 1) ###'
        analysis_string = self._get_grid_analysis_string(state, initial_grid_for_analysis)
        prompt_parts.extend([analysis_header, analysis_string])

        task_section = (
            '### TASK ###\n'
            'Based on the puzzle, the current state, and the grid analysis, determine the single most logical next DSL command. The command must be general enough to work for ALL training examples, not just the one shown.\n\n'
            'EXAMPLES OF PERFECT RESPONSE FORMATS:\n'
            'find_all_objects_by_color_connectivity()\n'
            'filter_selection_by_color(3)\n'
            'move_selected_objects_by_delta(0, 1)\n'
            'filter_selection_by_bounding_box(\'top_half\')\n\n'
            'Your response must be only the single line of code for the next command. Do not provide explanations or any other text.'
        )
        prompt_parts.append(task_section)

        return '\n\n'.join(prompt_parts)

    def _verify_state(self, state, expected_output_grids):
        if len(state.current_grids) != len(expected_output_grids):
            return False
        return all(torch.equal(g1, g2) for g1, g2 in zip(state.current_grids, expected_output_grids))

    def _grids_to_hash(self, grids):
        return hash(tuple(grid.numpy().tobytes() for grid in grids))