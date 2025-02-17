import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import backoff
import requests

from llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS
from smolagents import * 
from scripts.text_inspector_tool import TextInspectorTool
from scripts.visual_qa import visualizer
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]


user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)

parameter_search_system_message = """You are an ambitious AI, Economics and Complex Systems PhD student, who is looking to publish a paper that will contribute significantly to the field.
Previously you have searched for some ideas and judged whether an idea is novel or not. 
You will now be given access to the wider web, which you may use to further research the literature and find relevant parameters and information you can use for your Agent Based Modelling experiment.

{task_description}

<experiment.py>
{code}
</experiment.py>
"""


parameter_search_prompt = '''
Previousyly you came up with this idea and decided that it is a novel idea to implement and experiment with inside the Agent Based Modelling Environment.

"""
{idea}
"""
You have already done some research on the idea and found some relevant papers. 
However, you will also need several different sorts of agents to run your experiment in the given ABM environment. 

You are now tasked with finding the relevant agents and parameters for these agents. 

In <THOUGHT>, first briefly reason over the idea and identify the agents that could help you run your experiments.
Then, use the web to search for the relevant parameters for the agents you have identified. 
Ensure that your answers are correct and relevant to the idea and the agents you have identified. Also ensure that the parameters are relevant to the ABM environment (the environment is written in JAX). 

In <JSON>, respond in JSON format with ONLY the following field:
- "Agent_type": "The relevant parameters.", "What the agent does", "And the source of your identified parameters. "

This JSON will be automatically parsed, so ensure the format is precise.
'''

#we only do this once per idea, and not for a number of rounds because of the deep thinking 
def deep_research_data(
        base_dir,
        model_id,
        max_num_thinking_steps=20,
):
    with open(osp.join(base_dir, "ideas.json"), "r") as f:
        ideas = json.load(f)
        #agents = ideas["agents"]
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        idea_system_prompt = prompt["system"]
        task_description = prompt["task_description"]

    for idx, idea in enumerate(ideas):
        if "novel" not in idea:
            print(f"Skipping idea {idx}, not yet checked for novelty.")
            continue

        print(f"\nDeep research for idea {idx}: {idea['Name']}")

        
        system_message = parameter_search_system_message.format(
                    task_description=task_description,
                    code=code,
                )
        prompt = parameter_search_prompt.format(
                    idea=idea,
                    #agents=agents,
                )

        try:
            ## SEARCH FOR AGENT AND ENVIRONMENT PARAMETERS
            text_limit = 100000
        
            #have this work with the llm.py file. Could change it to use local llms from llm.py
            model = LiteLLMModel(
                model=model_id,
                custom_role_conversions={"tool-call": "assistant", 
                                         "tool-response": "user"},
                max_completion_tokens=8192,
                reasoning_effort="high",
                api_key=OPENAI_API_KEY,
                messages=[{"role": "system", "content": system_message}],
            )
            document_inspection_tool = TextInspectorTool(model, text_limit)

            browser = SimpleTextBrowser(**BROWSER_CONFIG)

            WEB_TOOLS = [
                SearchInformationTool(browser),
                VisitTool(browser),
                PageUpTool(browser),
                PageDownTool(browser),
                FinderTool(browser),
                FindNextTool(browser),
                ArchiveSearchTool(browser),
                TextInspectorTool(model, text_limit),
            ]

            text_webbrowser_agent = ToolCallingAgent(
                model=model,
                tools=WEB_TOOLS,
                max_steps=max_num_thinking_steps,
                verbosity_level=2,
                planning_interval=4,
                name="search_agent",
                description="""A team member that will search the internet to answer your question.
            Ask him for all your questions that require browsing the web.
            Provide him as much context as possible, in particular if you need to search on a specific timeframe!
            And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
            Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
            """,
                provide_run_summary=True,
            )
            text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
            If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
            Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

            manager_agent = CodeAgent(
                model=model,
                tools=[visualizer, document_inspection_tool],
                max_steps=max_num_thinking_steps,
                verbosity_level=2, #play around with these
                additional_authorized_imports=AUTHORIZED_IMPORTS,
                planning_interval=4, #play around with these
                managed_agents=[text_webbrowser_agent], #so far only managing one agent, can give more 
            )

            research_answer = manager_agent.run(prompt) #change this to prompt question 
            
        except Exception as e:
            print(f"Error: {e}")
            continue

    # Save results to JSON file
    research_results_file = osp.join(base_dir, "parameters.json")
    research_answer = {}
    with open(research_results_file, "w") as f:
        json.dump(research_answer, f, indent=4)

    return research_answer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate AI scientist parameters")
    args = parser.parse_args()
    
    current_path = '/Users/vercingetorix/Gone_fishing/Sakana5/'
    base_dir = osp.join(current_path, "templates/ai_economist")
    results_dir = osp.join("results", "abm")
    print(base_dir)
    answer = deep_research_data(
        base_dir=base_dir,
        #client="gpt3",
        model_id="o3-mini",
        max_num_thinking_steps=20
    )
 