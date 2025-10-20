import os
import random
import numpy as np
import torch
import model
import dsl
import utils
from search import NGPSSearch
import traceback


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def solve_task_chunk(args):
    task_chunk, gpu_id, results_dict, model_path, seed = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    set_seed(seed)

    dsl_executor = dsl.DSLExecutor()

    llm_guide = model.LLMGuide(
        model_path,
        seed=seed
    )

    arc_solver = ARCSolver(llm_guide, dsl_executor, utils)

    for _, task_id, task_data in task_chunk:
        print(f'[GPU {gpu_id}] Attempting to solve task: {task_id}')
        try:
            predictions = arc_solver.solve(task_data)
            results_dict[task_id] = predictions
            print(f'[GPU {gpu_id}] Successfully finished task: {task_id}')
        except Exception as e:
            print(f'[GPU {gpu_id}] CRITICAL ERROR on task {task_id}: {e}')
            print(traceback.format_exc())
            num_test_inputs = len(task_data.get('test', []))
            blank_predictions = [{'attempt_1': [], 'attempt_2': []} for _ in range(num_test_inputs)]
            results_dict[task_id] = blank_predictions

    return True


class ARCSolver:
    def __init__(self, llm_guide, dsl_executor, utils_module):
        self.llm_guide = llm_guide
        self.dsl_executor = dsl_executor
        self.utils = utils_module
        self.search_engine = NGPSSearch(
            llm_guide=llm_guide,
            dsl_executor=dsl_executor,
            utils_module=utils_module
        )
        self.dsl_definition = self._get_dsl_definition()

    def solve(self, task_json):
        training_pairs = task_json['train']
        base_prompt = self._build_base_prompt()

        program = self.search_engine.find_program_for_task(base_prompt, training_pairs)

        if not program:
            print('Search failed to find a program.')
            num_test_inputs = len(task_json.get('test', []))
            return [{'attempt_1': [], 'attempt_2': []} for _ in range(num_test_inputs)]

        print(f'Found solution program:\n{program}')

        solved_example = {
            'inputs': [p['input'] for p in training_pairs],
            'outputs': [p['output'] for p in training_pairs],
            'program': program
        }

        all_predictions = []
        for test_pair in task_json['test']:
            test_input_grid = test_pair['input']
            generalization_prompt = self._build_generalization_prompt([solved_example], test_input_grid)

            generated_programs = self.llm_guide.get_program_generation([generalization_prompt])

            if not generated_programs:
                all_predictions.append({'attempt_1': [], 'attempt_2': []})
                continue

            programs_to_try = [p.strip() for p in generated_programs[0].split('---') if p.strip()]

            attempt_1_grid = []
            if len(programs_to_try) >= 1:
                output_tensor_1 = self.dsl_executor.execute_script(programs_to_try[0], test_input_grid)
                if output_tensor_1 is not None:
                    attempt_1_grid = output_tensor_1.int().tolist()

            attempt_2_grid = []
            if len(programs_to_try) >= 2:
                output_tensor_2 = self.dsl_executor.execute_script(programs_to_try[1], test_input_grid)
                if output_tensor_2 is not None:
                    attempt_2_grid = output_tensor_2.int().tolist()

            prediction_pair = {
                'attempt_1': attempt_1_grid,
                'attempt_2': attempt_2_grid
            }
            all_predictions.append(prediction_pair)

        return all_predictions

    def _build_base_prompt(self):
        prompt_parts = [
            'INSTRUCTION: You are a reasoning engine that produces DSL commands to solve puzzles.',
            'Analyze the provided puzzle context and generate the single most promising next DSL command that works for ALL examples.',
            '### DOMAIN-SPECIFIC LANGUAGE (DSL) DEFINITION ###',
            'The following functions are available. The script must only use these functions.',
            self.dsl_definition
        ]
        return '\n\n'.join(prompt_parts)

    def _build_generalization_prompt(self, solved_examples, test_input_grid):
        prompt_parts = [
            'INSTRUCTION: You are a reasoning engine that generalizes from a solved puzzle to create a new DSL program.',
            'Analyze the solved example to deduce the underlying abstract logic.',
            'Apply this logic to the new test case and generate the specific DSL program that solves it.',
            '### DOMAIN-SPECIFIC LANGUAGE (DSL) DEFINITION ###',
            'The following functions are available.',
            self.dsl_definition,
            '### SOLVED EXAMPLE ###',
            'The following puzzle was solved with a single program that transforms every input grid to its corresponding output grid.'
        ]

        for example in solved_examples:
            prompt_parts.append('--- TASK EXAMPLES ---')
            for i in range(len(example['inputs'])):
                input_grid_str = self.utils.format_grid_to_string(example['inputs'][i])
                output_grid_str = self.utils.format_grid_to_string(example['outputs'][i])
                example_str = (
                    f'INPUT_GRID_{i+1}:\n\'{input_grid_str}\'\n\n'
                    f'OUTPUT_GRID_{i+1}:\n\'{output_grid_str}\''
                )
                prompt_parts.append(example_str)

            prompt_parts.append(f'\nSOLUTION_PROGRAM:\n{example["program"]}')

        test_input_str = self.utils.format_grid_to_string(test_input_grid)
        task_section = (
            '### TASK ###',
            'Generate the specific DSL program that solves the following test case:',
            f'TEST_INPUT_GRID:\n\'{test_input_str}\'',
            'Your output must be only the DSL program code. Provide up to two distinct, plausible programs, separated by "---". No other text or explanation is permitted.'
        )
        prompt_parts.extend(task_section)

        return '\n\n'.join(prompt_parts)

    def _get_dsl_definition(self):
        return '\n'.join(sorted(self.dsl_executor.get_operation_names()))