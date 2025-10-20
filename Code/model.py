from vllm import LLM, SamplingParams
import re


class LLMGuide:
    def __init__(self, model_path, context_length=16384, max_new_tokens=1024, tensor_parallel_size=1, seed=42):
        self.llm_engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            kv_cache_dtype='auto',
            dtype='bfloat16',
            enforce_eager=True,
            max_model_len=context_length,
            seed=seed
        )

        self.command_sampling_parameters = SamplingParams(
            temperature=0.1,
            top_p=0.7,
            max_tokens=32,
            stop=['\n']
        )

        self.program_sampling_parameters = SamplingParams(
            temperature=0.1,
            top_p=0.7,
            max_tokens=max_new_tokens
        )

    def _parse_dsl_command(self, raw_text):
        raw_text = raw_text.strip()
        match = re.search(r'(\w+\(.*\))', raw_text)
        if match:
            return match.group(1)

        cleaned_text = raw_text.strip('`\'" -*')
        if '(' in cleaned_text and ')' in cleaned_text:
            return cleaned_text

        return None

    def get_next_command(self, prompts, num_candidates=1):
        if not prompts:
            return []

        self.command_sampling_parameters.n = num_candidates

        outputs = self.llm_engine.generate(prompts, self.command_sampling_parameters)

        results = []
        for i in range(len(prompts)):
            prompt_outputs = outputs[i]
            candidates = [self._parse_dsl_command(candidate.text) for candidate in prompt_outputs.outputs]
            results.append([c for c in candidates if c is not None])

        return results

    def get_program_generation(self, prompts):
        if not prompts:
            return []

        outputs = self.llm_engine.generate(prompts, self.program_sampling_parameters)
        generated_scripts = [output.outputs[0].text.strip() for output in outputs if output.outputs]
        return generated_scripts