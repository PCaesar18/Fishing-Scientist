#check up on https://github.com/ShengranHu/ADAS

#in this file, we will generate (RL) agents for use in our (JAX) Environment 
# like the Automated Design of Agentic Systems paper, the goal is also to create an agent archive for further reference

#the agents will be run in the environment to set an initial baseline and to see if they work properly


#also need to create an archive for emergent behaviour/properties

import argparse
import copy
import json
import os
import pickle
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm
from templates.ai_economist.environment.agents.base_agent import BaseAgent

from utils import random_id,list_to_string, get_prompt, get_init_archive




ROLE_DESC = lambda role: f"You are a {role}.\n\n"
SYSTEM_MSG = ""
CODE_INST = "You will write code to solve this task by creating a function named `transform`. This function should take a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation that works for both example and test inputs. Make sure that the transform function is capable of handling both example and test inputs effectively, reflecting the learned transformation rules from the Examples inputs and outputs."

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


base_prompt = """# Overview
You are an expert machine learning researcher testing out various reinforcement learning Agents. 
Your objective is to design Economic RL agents in JAX, with specific building blocks such as control flows and specific hyperparemeters within these systems to solve complex tasks. 
Your aim is to design an optimal agent performing well in the economic environment that it is operating in. Feedback will be provided based on the performance of the agent in the environment.

## Type of RL agent you have to create and will be used in the future 
{AGENT_TYPE}

## Agent Hyperparameters 

{HYPERPARAMETERS}

## The base Agent Class which you have to inherit from and adapt :

{BASE_AGENT_CLASS}

# The environment code:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
client = openai.OpenAI()

# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for LLM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\n"

# Description of the role for the LLM
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    \"""
    Function to get JSON response from GPT model.
    
    Args:
    - msg (str): The user message.
    - model (str): The model to use.
    - system_message (str): The system message.
    - temperature (float): Sampling temperature.
    
    Returns:
    - dict: The JSON response.
    \"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase:
    \"""
    Base class for an LLM agent.
    
    Attributes:
    - output_fields (list): Fields expected in the output.
    - agent_name (str): Name of the agent.
    - role (str): Role description for the agent.
    - model (str): Model to be used. (option. Keep it default.)
    - temperature (float): Sampling temperature.
    - id (str): Unique identifier for the agent instance.
    \"""

    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()
    
    def generate_prompt(self, input_infos, instruction) -> str:
        \"""
        Generates a prompt for the LLM.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        
        Returns:
        - tuple: System prompt and user prompt.

        An example of a generated prompt:
        ""
        You are a helpful assistant.
        
        # Output Format:
        Reply EXACTLY with the following JSON format.
        ...

        # Your Task:
        You will be given some number of paired example inputs and outputs. The outputs ...

        ### thinking #1 by Chain-of-Thought Agent hkFo (yourself):
        ...
        
        ### code #1 by Chain-of-Thought Agent hkFo (yourself):
        ...

        ### answer by Chain-of-Thought Agent hkFo's code evaluator:...


        # Instruction: 
        Please think step by step and then solve the task by writing the code.
        ""
        \"""
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt 

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        \"""
        Queries the LLM with provided input information and instruction.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        - iteration_idx (int): Iteration index for the task.
        
        Returns:
        - output_infos (list[Info]): Output information.
        \"""
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"
    
    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        # Note:
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    \"""
    Fill in your code here.
    \"""
    def forward(self, taskInfo) -> Union[Info, str]:
        \"""
        Placeholder method for processing task information.
        
        Args:
        - taskInfo (Info): Task information.
        
        Returns:
        - Answer (Union[Info, str]): Your FINAL Answer. Return either a namedtuple Info or a string of answers.
        \"""
        pass
```
# Previousyly discovered agent architecture archive
Here is the archive of the discovered agent architectures that worked well in the environment. You can use these as inspiration for your new agent design or use them in your new design:

{ARCHIVE}

The fitness value is the median and 95% Bootstrap Confidence Interval of the correct rate on a validation question set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture. 
Finally, the last key ("code") corresponds to the exact .“forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

{EXAMPLE}

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. 
Also, it could be helpful to set the LLM’s role and temperature to further control the LLM’s response. Note that the LLMAgentBase() will automatically parse the output and return a list of “Infos”. You can get the content by Infos.content. 
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.

## WRONG Implementation examples:
Here are some mistakes you may make:

1. This is WRONG: ```
feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
```
It is wrong to use "Info('feedback', 'Critic Agent', thinking, 0)". The returned "feedback" from LLMAgentBase is already Info.

2. This is WRONG: ```
# Debugging: Log the generated answer
print('Generated Answer:', ...)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'
```
First, the len(feedback_info) will not work.
Second, you should never return an error message. You should always return the best answer you can get.
Third, you should never print anything in the code.
Lastly, again, DO NOT CREATE Info object by yourself.

3. This is WRONG: ```
all_thinking = []
all_answers = []
for agent, role in zip(agents, roles):
    outputs = agent([taskInfo], independent_reasoning_instruction.format(role=role))
    all_thinking.append(outputs[0].content)
    all_answers.append(outputs[1].content)

# Aggregate the reasoning paths and answers
aggregated_thinking = '\n'.join(all_thinking)
aggregated_answers = '\n'.join(all_answers)
```
You SHOULD NOT extract the content from the Info object by yourself. You should use the Info object directly. If you want to aggregate the content, you should just put those Info objects into a list and then use the list as input to the next LLM agent.

4. This is WRONG: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
response_infos = reasoning_agent([taskInfo] + ..., reasoning_instruction)
    
# Extract the final answer from the response_infos
for info in response_infos:
    if info.name == 'final_answer':
        return info
# Fallback if no answer is found
return Info('answer', 'Final Decision Agent', 'No answer generated.', 0)
```
You should not extract the final answer by yourself. You SHOULD directly return the answer Info. Also, you should always return the best answer you can get.
CORRECT example: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = reasoning_agent([taskInfo] + ..., reasoning_instruction)
return answer
```

# Your task
You are deeply familiar with reinforcement techniques and the agent works from the literature. Your goal is to maximize the specified performance metrics by proposing interestingly new RL agents.
Observe the discovered agents carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative when thinking about the next interesting agent to try. You are encouraged to draw inspiration from related agent papers or academic papers from other research areas.
Use the knowledge from the archive and inspiration from the hyperparameters and academic literature to propose the next interesting RL agentic system.
THINK OUTSIDE THE BOX.
"""


iteration_prompt_1 = f""""[EXAMPLE]Carefully review the proposed new agent architecture and reflect on the following points:"

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings. 
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""


# @backoff.on_exception(backoff.expo, openai.RateLimitError)
# def get_json_response_from_gpt(
#         msg,
#         model,
#         system_message,
#         temperature=0.5
# ):
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_message},
#             {"role": "user", "content": msg},
#         ],
#         temperature=temperature, max_tokens=1024, stop=None, response_format={"type": "json_object"}
#     )
#     content = response.choices[0].message.content
#     json_dict = json.loads(content)
#     # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
#     assert not json_dict is None
#     return json_dict


# @backoff.on_exception(backoff.expo, openai.RateLimitError)
# def get_json_response_from_gpt_reflect(
#         msg_list,
#         model,
#         temperature=0.8
# ):
#     response = client.chat.completions.create(
#         model=model,
#         messages=msg_list,
#         temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
#     )
#     content = response.choices[0].message.content
#     json_dict = json.loads(content)
#     assert not json_dict is None
#     return json_dict


#fix this to create full prompts for input fields


def get_prompt(idea, current_archive, base_agent, parameters):
    """
    Constructs and returns a system prompt and a user prompt for generating new agent architectures.

    This function takes in details about the agent type, current archive of solutions, base agent class, 
    and hyperparameters, and fills in a template prompt with these details. The prompt is used to guide 
    the generation of new agent architectures in a reinforcement learning environment.

    Args:
        idea (str): The type of RL agent to be created and used in the future.
        current_archive (list): A list of current solutions or agents, each represented as a dictionary.
        base_agent (str): The base agent class which the new agent should inherit from and adapt.
        parameters (dict): A dictionary of hyperparameters for the agent.

    Returns:
        tuple: A tuple containing the system prompt and the filled user prompt.
    """
    
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"

    
    prompt = base_prompt.replace("{ARCHIVE}", archive_str)
    prompt = prompt.replace("{AGENT_TYPE}", idea)
    prompt = prompt.replace("{HYPERPARAMETERS}", json.dumps(parameters))
    prompt = prompt.replace("{BASE_AGENT_CLASS}", base_agent)
    prompt = prompt.replace("{EXAMPLE}", json.dumps(EXAMPLE)) #previous example of created agent 

    # Return the system prompt and the filled prompt
    return system_prompt, prompt


def get_init_archive():
    return [government_agent, population_agent]


def get_iteration_prompt(prev_example):
    """
    Constructs and returns a pair of reflection prompts for evaluating and iterating on agent designs.

    This function takes a previous example of an agent and integrates it into a reflection prompt template.
    The purpose of these prompts is to guide the user or system in reflecting on the previous agent's design,
    identifying areas for improvement, and proposing new iterations or enhancements.

    Args:
        prev_example (dict): A dictionary representing the previous agent example, which includes details
                             about the agent's design and performance.

    Returns:
        tuple: A tuple containing two strings:
            - The first string is the reflection prompt with the previous example integrated.
            - The second string is a static reflection prompt template for further guidance.
    """
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    r1 = iteration_prompt_1.replace("[EXAMPLE]", prev_example_str) if prev_example else iteration_prompt_1.replace("[EXAMPLE]", "")
    return r1, Reflexion_prompt_2 #TODO: iteration prompt 2 does not exist yet (is it necessary?)


def generate_prompt(self, input_infos, instruction) -> str:
    code_output = False

    # construct system prompt
    output_fields_and_description = {key: f"Your {key}." for key in self.output_fields}
    for key in output_fields_and_description:
        if 'answer' in key:
            output_fields_and_description[key] = f"Your {key}. ONLY return a string of list[list[int]]. DO NOT return anything else."
        elif 'code' in key:
            output_fields_and_description[key] = f"Your {key}. Don't write tests in your Python code, ONLY return the `transform` function. DO NOT return anything else. (It will be tested later.)"
            code_output = True
    system_prompt = ROLE_DESC(self.role) + FORMAT_INST(output_fields_and_description)

    # construct input infos text
    input_infos_text = ''
    for input_info in input_infos:
        if isinstance(input_info, Info):
            (field_name, author, content, iteration_idx) = input_info
        else:
            continue

        if isinstance(content, list):
            try:
                content = list_to_string(content)
            except:
                pass

        if author == self.__repr__():
            author += ' (yourself)'
        if field_name == 'task':
            input_infos_text += f'# Your Task:\n{content}\n\n'
        elif iteration_idx != -1:
            input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
        else:
            input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

    prompt = input_infos_text + "# Instruction: \n" + instruction + "\n\n" + (CODE_INST if code_output else '')
    return system_prompt, prompt

def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
    system_prompt, prompt = self.generate_prompt(input_infos, instruction)
    try:
        response_json = {}
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature) #here we get the response from our LLM 
        assert len(response_json) == len(self.output_fields), "not returning enough fields"
    except Exception as e:
        # print(e)
        if "maximum context length" in str(e) and SEARCHING_MODE:
            raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
        # try to fill in the missing field
        for key in self.output_fields:
            if not key in response_json and len(response_json) < len(self.output_fields):
                response_json[key] = ''
        for key in copy.deepcopy(list(response_json.keys())):
            if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                del response_json[key]
    output_infos = []
    for key, value in response_json.items():
        info = Info(key, self.__repr__(), value, iteration_idx)
        output_infos.append(info)
    return output_infos

def __repr__(self):
    return f"{self.agent_name} {self.id}"

def __call__(self, input_infos: list, instruction, iteration_idx=-1):
    return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self, examples, test_iuput) -> None:
        self.examples = examples
        self.test_iuput = test_iuput

    def run_examples_and_get_feedback(self, code):
        examples = self.examples

        correct_examples = []
        wrong_examples = []

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('feedback', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}"), correct_examples, wrong_examples
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code."), correct_examples, wrong_examples

        transform = local_vars['transform']

        feedback = ""

        for idx, example in enumerate(examples):
            input_grid = example['input']
            output_grid = example['output']
            try:
                transformed_grid = transform(input_grid)
            except Exception as e:
                return gen_output(f"Error during function execution: {e}"), correct_examples, wrong_examples

            if transformed_grid == output_grid:
                feedback += f"Your transform function generates a CORRECT answer in Example {idx}!\n\n"
                correct_examples.append(example)
            else:
                try:
                    transformed_grid = list_to_string(transformed_grid)
                except:
                    pass
                feedback += f"Your transform function generates a WRONG answer in Example {idx}!\nExpect: See above Example {idx} output.\nYou got: {transformed_grid}\nObserve the Example {idx} carefully!\n\n"
                wrong_examples.append(example)

        return gen_output(feedback), correct_examples, wrong_examples

    def get_test_output_from_code(self, code):
        test_input = self.test_iuput

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('answer', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}")
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code.")

        transform = local_vars['transform']
        try:
            transform_output = transform(test_input)
            transform_output = list_to_string(transform_output)
        except Exception as e:
            return gen_output(f"Error during function execution: {e}")

        return gen_output(transform_output)


def gen_agents(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)

            Reflexion_prompt_1, Reflexion_prompt_2 = get_iteration_prompt(archive[-1] if n > 0 else None)
            # Reflexion 1
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            # Reflexion 2
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            continue

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        if not acc_list:
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def eval_agents(args):
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        try:
            acc_list = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)

        current_idx += 1


def evaluate_forward_fn(args, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    if SEARCHING_MODE:
        arc_dir = args.val_data_path
    else:
        arc_dir = args.test_data_path
    print(arc_dir)
    with open(arc_dir, 'rb') as pickle_file:
        arc_data_queue = pickle.load(pickle_file)

    print(f"problem length: {len(arc_data_queue) * args.n_repreat}")
    max_workers = min(len(arc_data_queue) * args.n_repreat, args.max_workers) if args.multiprocessing else 1

    agent_task_queue = []
    for arc_data in arc_data_queue:
        task_str, examples, test_input = format_arc_data(arc_data)
        taskInfo = Info('task', 'User', task_str, -1)
        agent_task_queue.extend([(AgentSystem(examples, test_input), taskInfo, arc_data)] * args.n_repreat)

    def call_forward(agent_task_queue):
        agent, taskInfo, arc_data = agent_task_queue
        res = agent.forward(taskInfo)
        origin_res = res
        try:
            if isinstance(res, Info):
                res = res.content
            if isinstance(res, str):
                res = eval(res)
            hard_score = eval_solution(res, arc_data, soft_eval=False)
            return hard_score
        except Exception as e:
            # print(e)
            return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        acc_list = list(tqdm(executor.map(call_forward, agent_task_queue), total=len(agent_task_queue)))

    print("acc:", bootstrap_confidence_interval(acc_list))
    return acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data_path', type=str, default='sampled_arc_val_data.pkl')
    parser.add_argument('--test_data_path', type=str, default='sampled_arc_test_data.pkl')
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default='arc_gpt3.5_results')
    parser.add_argument('--n_generation', type=int, default=25)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4o-2024-05-13',
                        choices=['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13'])

    args = parser.parse_args()
    # search
    SEARCHING_MODE = True
    gen_agents(args)

    # evaluate
    SEARCHING_MODE = False
    eval_agents(args)