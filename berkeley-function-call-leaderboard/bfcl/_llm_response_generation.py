import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy

import numpy as np
from tqdm import tqdm

from bfcl._apply_function_credential_config import apply_function_credential_config
from bfcl.constant import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    MULTI_TURN_FUNC_DOC_PATH,
    PROJECT_ROOT,
    PROMPT_PATH,
    RESULT_PATH,
    TEST_FILE_MAPPING,
    TEST_IDS_TO_GENERATE_PATH,
)
from bfcl.eval_checker.eval_runner_helper import load_file
from bfcl.model_handler.handler_map import HANDLER_MAP
from bfcl.model_handler.model_style import ModelStyle
from bfcl.utils import (
    check_api_key_supplied,
    is_executable,
    is_multi_turn,
    is_nestful,
    parse_test_category_argument,
    sort_key,
)

RETRY_LIMIT = 30
# 60s for the timer to complete. But often we find that even with 60 there is a conflict. So 65 is a safe no.
RETRY_DELAY = 65  # Delay in seconds

# Initialise logger
logger = logging.getLogger("response_generation")


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="gorilla-openfunctions-v2", nargs="+")
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default="all", nargs="+")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--include-input-log", action="store_true", default=False)
    parser.add_argument("--exclude-state-log", action="store_true", default=False)
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--backend", default="vllm", type=str, choices=["vllm", "sglang"])
    parser.add_argument("--gpu-memory-utilization", default=0.9, type=float)
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--run-ids", action="store_true", default=False)
    parser.add_argument("--allow-overwrite", "-o", action="store_true", default=False)
    # Add the new skip_vllm argument
    parser.add_argument(
        "--skip-server-setup",
        action="store_true",
        default=False,
        help="Skip vLLM/SGLang server setup and use existing endpoint specified by the VLLM_ENDPOINT and VLLM_PORT environment variables.",
    )
    args = parser.parse_args()
    return args


def build_handler(model_name, temperature):
    handler = HANDLER_MAP[model_name](model_name, temperature)
    return handler


def get_involved_test_entries(test_category_args, run_ids):
    all_test_file_paths, all_test_categories, all_test_entries_involved = [], [], []
    api_key_supplied = check_api_key_supplied()
    skipped_categories = []
    if run_ids:
        with open(TEST_IDS_TO_GENERATE_PATH) as f:
            test_ids_to_generate = json.load(f)
        for category, test_ids in test_ids_to_generate.items():
            if len(test_ids) == 0:
                continue
            test_file_path = TEST_FILE_MAPPING[category]
            all_test_entries_involved.extend([entry for entry in load_file(PROMPT_PATH / test_file_path) if entry["id"] in test_ids])
            # Skip executable test category if api key is not provided in the .env file
            if (is_executable(category) and not api_key_supplied) and not is_nestful(category):
                skipped_categories.append(category)
            else:
                all_test_categories.append(category)
                all_test_file_paths.append(test_file_path)

    else:
        all_test_file_paths, all_test_categories = parse_test_category_argument(test_category_args)
        # Make a copy here since we are removing list elemenets inside the for loop
        for test_category, file_to_open in zip(all_test_categories[:], all_test_file_paths[:]):
            if (is_executable(test_category) and not api_key_supplied) and not is_nestful(test_category):
                all_test_categories.remove(test_category)
                all_test_file_paths.remove(file_to_open)
                skipped_categories.append(test_category)
            else:
                all_test_entries_involved.extend(load_file(PROMPT_PATH / file_to_open))

    return (
        all_test_file_paths,
        all_test_categories,
        all_test_entries_involved,
        skipped_categories,
    )


def collect_test_cases(args, model_name, all_test_categories, all_test_file_paths, all_test_entries_involved):
    model_name_dir = model_name.replace("/", "_")
    model_result_dir = args.result_dir / model_name_dir

    existing_result = []
    for test_category, file_to_open in zip(all_test_categories, all_test_file_paths):

        result_file_path = model_result_dir / file_to_open.replace(".json", "_result.json")
        if result_file_path.exists():
            # Not allowing overwrite, we will load the existing results
            if not args.allow_overwrite:
                existing_result.extend(load_file(result_file_path))
            # Allow overwrite and not running specific test ids, we will delete the existing result file before generating new results
            elif not args.run_ids:
                result_file_path.unlink()
            # Allow overwrite and running specific test ids, we will do nothing here
            else:
                pass

        existing_ids = [entry["id"] for entry in existing_result]

    test_cases_to_generate = [test_case for test_case in all_test_entries_involved if test_case["id"] not in existing_ids]
    test_cases_to_generate = process_multi_turn_test_case(test_cases_to_generate)

    return sorted(test_cases_to_generate, key=sort_key)


def process_multi_turn_test_case(test_cases):
    """
    Multi-turn test cases don't have the function doc in the prompt. We need to add them here.
    """
    # Load the global functions dataset in case functions need to be added
    global_functions_jsonl_path = PROMPT_PATH / "global_functions.jsonl"
    global_functions = load_file(global_functions_jsonl_path)

    for entry in test_cases:
        if not is_multi_turn(entry["id"]):
            continue
        involved_classes = entry["involved_classes"]
        entry["function"] = []
        func_docs = {}
        for func_collection in involved_classes:
            # func_doc is a list of dict
            func_doc = load_file(MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection])
            func_docs[func_collection] = func_doc
            entry["function"].extend(func_doc)

        # Check if the entry is of the "_128tools" category
        if "_128tools" in entry["id"]:
            # Verify if the number of functions is not > 128
            # If it is, take out the last functions while
            # making sure they are not functions of the
            # "involved_classes_original" list
            if len(entry["function"]) > 128:
                while len(entry["function"]) > 128:
                    # Take the last entry
                    last_entry = entry["function"].pop()
                    # Iterate the function collection names
                    for func_collection_name, func_collection_docs in func_docs.items():

                        # If the function collection is part of the original involved classes, skip
                        if func_collection_name in entry["involved_classes_original"]:
                            # If the last entry is part of the original involved classes, add it back
                            # (to the beginning so we don't pop it again)
                            if last_entry in func_collection_docs:
                                entry["function"].insert(0, last_entry)
                            # Skip (continue) to the next function collection
                            continue

                        # If the last entry is not part of the original involved classes, remove it
                        if last_entry in func_collection_docs:
                            break

        # Check if the entry is of the "_900tools" category
        if "_900tools" in entry["id"]:
            target_num_functions = 900
            if len(entry["function"]) >= target_num_functions:
                entry["function"] = entry["function"][:target_num_functions]
            else:
                print(f"Test case ID: {entry['id']}, Number of functions was {len(entry['function'])}", end=" ")
                remaining_functions = target_num_functions - len(entry["function"])
                # Check if the entry is holding duplicate functions when replacing "." with "_"
                # If it is, remove the duplicates
                entry_function_names = [copy(func["name"]).replace(".", "_") for func in entry["function"]]
                if len(entry_function_names) != len(set(entry_function_names)):
                    print("INITIAL - Entry function names are not unique.")
                    # Print equal names
                    for name in set([name for name in entry_function_names if entry_function_names.count(name) > 1]):
                        print(f"Equal function names: {name}")
                    # Show original function names
                    for func in entry["function"]:
                        if copy(func["name"]).replace(".", "_") in set([name for name in entry_function_names if entry_function_names.count(name) > 1]):
                            print(f"Original function name: {func['name']}")
                    # Remove duplicates
                    entry["function"] = [func for func in entry["function"] if entry_function_names.count(copy(func["name"]).replace(".", "_")) == 1]
                    print(f"Removed duplicates. Now is {len(entry['function'])} functions.")

                while remaining_functions:
                    # Sample the functions from the global functions dataset
                    sampled_functions = np.random.choice(global_functions, remaining_functions, replace=False).tolist()
                    # Check if there aren't any duplicate names in the sample functions when replacing "." with "_"
                    sampled_function_names = [copy(func["name"]).replace(".", "_") for func in sampled_functions]
                    if len(sampled_function_names) != len(set(sampled_function_names)):
                        # Remove duplicates
                        sampled_functions = [func for func in sampled_functions if sampled_function_names.count(copy(func["name"]).replace(".", "_")) == 1]
                        # print(f"Removed duplicates. Now is {len(sampled_functions)} functions.")
                    # Check that there are no duplicate names between the sampled functions and the entry functions
                    if len(set(sampled_function_names).intersection(set(entry_function_names))) > 0:
                        # Remove duplicates
                        sampled_functions = [func for func in sampled_functions if copy(func["name"]).replace(".", "_") not in set(entry_function_names)]
                        # print(f"Removed duplicates. Now is {len(sampled_functions)} functions.")
                    # Add the sampled functions to the entry functions
                    entry["function"].extend(sampled_functions)
                    remaining_functions = target_num_functions - len(entry["function"])
                    # print(f"Added {len(sampled_functions)} functions. {remaining_functions} remaining.", end=" ")

                    # NOTE just a double check:
                    # Check if there aren't any duplicate names in the entry functions when replacing "." with "_"
                    entry_function_names = [copy(func["name"]).replace(".", "_") for func in entry["function"]]
                    if len(entry_function_names) != len(set(entry_function_names)):
                        print("DOUBLE CHECK - Entry function names are not unique.")
                        # Print equal names
                        for name in set([name for name in entry_function_names if entry_function_names.count(name) > 1]):
                            print(f"Equal function names: {name}")
                        # Show original function names
                        for func in entry["function"]:
                            if copy(func["name"]).replace(".", "_") in set([name for name in entry_function_names if entry_function_names.count(name) > 1]):
                                print(f"Original function name: {func['name']}")

                print(f"and now is {len(entry['function'])} functions.")

                # Check that all tool names are unique if replacing "." with "_"
                tool_names = [tool["name"].replace(".", "_") for tool in entry["function"]]
                if len(tool_names) != len(set(tool_names)):
                    print("Tool names are not unique.")
                # Print equal names
                for name in set([name for name in tool_names if tool_names.count(name) > 1]):
                    print(f"Equal tool names: {name}")
                # Show original tool names
                for tool in entry["function"]:
                    if tool["name"].replace(".", "_") in set([name for name in tool_names if tool_names.count(name) > 1]):
                        print(f"Original tool name: {tool['name']}")

        # Shuffle the functions
        entry["function"] = np.random.permutation(entry["function"]).tolist()

        # Handle Miss Func category; we need to remove the holdout function doc
        if "missed_function" in entry:
            for turn_index, missed_func_names in entry["missed_function"].items():
                entry["missed_function"][turn_index] = []
                for missed_func_name in missed_func_names:
                    for i, func_doc in enumerate(entry["function"]):
                        if func_doc["name"] == missed_func_name:
                            # Add the missed function doc to the missed_function list
                            entry["missed_function"][turn_index].append(func_doc)
                            # Remove it from the function list
                            entry["function"].pop(i)
                            break

    return test_cases


def multi_threaded_inference(handler, test_case, include_input_log, exclude_state_log):

    assert type(test_case["function"]) is list

    retry_count = 0

    logger.info(f"Generating result for test case: {test_case['id']}")

    while True:
        try:
            result, metadata = handler.inference(deepcopy(test_case), include_input_log, exclude_state_log)
            print(f"✅ Test case ID: {test_case['id']} - Success")
            logger.info(f"✅ Test case ID: {test_case['id']} - Success")
            break  # Success, exit the loop
        except Exception as e:
            # TODO: It might be better to handle the exception in the handler itself rather than a universal catch block here, as each handler use different ways to call the endpoint.
            # OpenAI has openai.RateLimitError while Anthropic has anthropic.RateLimitError. It would be more robust in the long run.
            if retry_count < RETRY_LIMIT and ("rate limit reached" in str(e).lower() or (hasattr(e, "status_code") and (e.status_code in {429, 503, 500}))):
                print(f"❗️ Rate limit reached. Sleeping for {RETRY_DELAY} seconds. Retry {retry_count + 1}/{RETRY_LIMIT}")
                logger.warning(
                    f"❗️ Rate limit reached. Sleeping for {RETRY_DELAY} seconds. Test case ID: {test_case['id']}, Error: {str(e)}. Retry {retry_count + 1}/{RETRY_LIMIT}"
                )
                time.sleep(RETRY_DELAY)
                # NOTE let's skip to increment the retry counter if the error is an ASI 'something went wrong'
                if "something went wrong" not in str(e).lower():
                    retry_count += 1
            else:
                # This is usually the case when the model getting stuck on one particular test case.
                # For example, timeout error or FC model returning invalid JSON response.
                # Since temperature is already set to 0.001, retrying the same test case will not help.
                # So we continue the generation process and record the error message as the model response
                print("-" * 100)
                print("❗️❗️ Error occurred during inference. Maximum retries reached for rate limit or other error. Continuing to next test case.")
                print(f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}")
                # Iterate through the tools in the test case and check if they respect the regex pattern
                import re

                for tool in test_case["function"]:
                    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", tool["name"]):
                        print(f"Invalid tool name: {tool['name']}")
                    # Check properties
                    for prop in tool["parameters"]["properties"]:
                        if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", prop):
                            print(f"Invalid property name: {prop}")
                    # Check required field
                    for req in tool["parameters"]["required"]:
                        if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", req):
                            print(f"Invalid required field name: {req}")
                # Check that all tool names are unique if replacing "." with "_"
                tool_names = [tool["name"].replace(".", "_") for tool in test_case["function"]]
                if len(tool_names) != len(set(tool_names)):
                    print("Tool names are not unique.")
                # Print equal names
                for name in set([name for name in tool_names if tool_names.count(name) > 1]):
                    print(f"Equal tool names: {name}")
                # Show original tool names
                for tool in test_case["function"]:
                    if tool["name"].replace(".", "_") in set([name for name in tool_names if tool_names.count(name) > 1]):
                        print(f"Original tool name: {tool['name']}")

                print("-" * 100)
                logger.error(
                    f"❗️❗️❗️ Error occurred during inference. Maximum retries reached for rate limit or other error. Continuing to next test case. Test case ID: {test_case['id']}, Error: {str(e)}"
                )

                return {
                    "id": test_case["id"],
                    "result": f"Error during inference: {str(e)}",
                }

    result_to_write = {
        "id": test_case["id"],
        "result": result,
    }

    result_to_write.update(metadata)

    return result_to_write


def generate_results(args, model_name, test_cases_total):
    update_mode = args.allow_overwrite
    handler = build_handler(model_name, args.temperature)

    if handler.model_style == ModelStyle.OSSMODEL:
        # batch_inference will handle the writing of results
        handler.batch_inference(
            test_entries=test_cases_total,
            num_gpus=args.num_gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            backend=args.backend,
            skip_server_setup=args.skip_server_setup,
            include_input_log=args.include_input_log,
            exclude_state_log=args.exclude_state_log,
            result_dir=args.result_dir,
            update_mode=update_mode,
        )

    else:
        futures = []
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            with tqdm(total=len(test_cases_total), desc=f"Generating results for {model_name}") as pbar:

                for test_case in test_cases_total:
                    future = executor.submit(
                        multi_threaded_inference,
                        handler,
                        test_case,
                        args.include_input_log,
                        args.exclude_state_log,
                    )
                    futures.append(future)

                for future in futures:
                    # This will wait for the task to complete, so that we are always writing in order
                    result = future.result()
                    handler.write(
                        result, result_dir=args.result_dir, update_mode=args.run_ids
                    )  # Only when we run specific test ids, we will need update_mode=True to keep entries in the same order
                    pbar.update()


def main(args):

    if type(args.model) is not list:
        args.model = [args.model]
    if type(args.test_category) is not list:
        args.test_category = [args.test_category]

    (
        all_test_file_paths,
        all_test_categories,
        all_test_entries_involved,
        skipped_categories,
    ) = get_involved_test_entries(args.test_category, args.run_ids)

    print(f"Generating results for {args.model}")
    if args.run_ids:
        print("Running specific test cases. Ignoring `--test-category` argument.")
    else:
        print(f"Running full test cases for categories: {all_test_categories}.")

    if len(skipped_categories) > 0:
        print("----------")
        print(
            f"❗️ Note: The following executable test category entries will be skipped because they require API Keys to be provided in the .env file: {skipped_categories}.\n Please refer to the README.md 'API Keys for Executable Test Categories' section for details.\n The model response for other categories will still be generated."
        )
        print("----------")

    # Apply function credential config if any of the test categories are executable
    # We can know for sure that any executable categories will not be included if the API Keys are not supplied.
    # if any([is_executable(category) for category in all_test_categories]):
    #     apply_function_credential_config(input_path=PROMPT_PATH)
    for category in all_test_categories:
        if is_executable(category) and not is_nestful(category):
            apply_function_credential_config(input_path=PROMPT_PATH / TEST_FILE_MAPPING[category])

    if args.result_dir is not None:
        args.result_dir = PROJECT_ROOT / args.result_dir
    else:
        args.result_dir = RESULT_PATH

    # Set logger
    root_logger = logging.getLogger("response_generation")
    file_handler = logging.FileHandler(f"{args.result_dir}/results.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    for model_name in args.model:
        test_cases_total = collect_test_cases(
            args,
            model_name,
            all_test_categories,
            all_test_file_paths,
            all_test_entries_involved,
        )

        if len(test_cases_total) == 0:
            print(f"All selected test cases have been previously generated for model {model_name}. No new test cases to generate.")
            logger.info(f"All selected test cases have been previously generated for model {model_name}. No new test cases to generate.")
        else:
            print(f"Generating results for model {model_name} with {len(test_cases_total)} test cases.")
            logger.info(f"Generating results for model {model_name} with {len(test_cases_total)} test cases.")
            generate_results(args, model_name, test_cases_total)
