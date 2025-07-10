import argparse
import csv
import json
import logging
import os
import pandas as pd
import re

from typing import Optional, Tuple, List, Dict

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def parse_arguments():
    """
    Parse command line arguments for the model evaluation script.
    Returns:
        argparse.Namespace: Parsed arguments containing evaluation configuration
    """
    parser = argparse.ArgumentParser(description='Evaluate model accuracy for llava models.')
    parser.add_argument('--mode', choices=['tqa', 'vqa', 'vtqa'], default='tqa')
    parser.add_argument('--output_folder', type=str, default='outputs/', help='Path to the directory containing model outputs.')
    parser.add_argument('--dataset_id', type=str, default='MilaWang/SpatialEval', help='Dataset identifier for Hugging Face.')
    parser.add_argument('--eval_summary_dir', type=str, default='eval_summary', help='Path to the directory to save evaluation summaries.')
    parser.add_argument('--task', type=str, default='spatialgrid', choices=['all', 'spatialmap', 'mazenav', 'spatialgrid', 'spatialreal'], help='Task to evaluate.')
    args = parser.parse_args()

    return args


def extract_model_name(filename: str, suffix: str = "_w_reason.jsonl") -> Optional[str]:
    """Extracts the model name from the filename."""
    prefix = "m-"
    if filename.startswith(prefix) and filename.endswith(suffix):
        return filename[len(prefix):-len(suffix)]
    return None


def extract_available_options(prompt):
    """
    Extract multiple choice options from a prompt text.
    Args:
        prompt (str): The prompt text
    
    Returns:
        list: List of tuples containing (letter, option_text) pairs
              e.g., [('A', 'North'), ('B', 'South'), ('C', 'East'), ('D', 'West')]
    """
    # Match the full block starting with "Available options:"
    # The pattern captures multiple lines of choices in format "A. Option text"
    block_pattern = r"Available options:\n((?:\s*[A-Z]\s*\.\s*.*\n?\s*)+)" # choice fix
    
    block_match = re.search(block_pattern, prompt)
    choices = []
    if block_match:
        block_text = block_match.group(1)
        # Extract each individual choice with letter and text
        # Pattern matches: "A. Some option text"
        choice_pattern = r'([A-Z])\s*\.\s*(.*)\s*'
        choices = re.findall(choice_pattern, block_text)
    else:
        logging.info("No match found when extracting `Available Options`")
    
    return choices


def extract_answer(json_string, choices):
    """
    Extract the model's answer from its response text.
    
    This function handles multiple response formats:
    1. Direct letter answers: "Answer: A"
    2. Text answers that match choice options: "Answer: North"
    3. Answers without explicit "Answer:" tag
    
    Args:
        json_string (str): The model's complete response text
        choices (list): List of available choice options from extract_available_options()
    
    Returns:
        tuple: (answer_letter, answer_text, is_direct_answer)
            - answer_letter: The extracted choice letter (A, B, C, etc.)
            - answer_text: The extracted answer text
            - is_direct_answer: Index if model gave direct text match, None otherwise
    """
    
    # Find choice letters A, B, C etc and corresponding text for that letter.
    letters = [x[0] for x in choices] # Example: ['A', 'B', 'C', 'D']
    choice_str = [re.sub(r'[^\w\s]|[\s]+$', '', x[1]) for x in choices]  # remove all trailing space and punctuation
    choice_str_lower = [x.lower() for x in choice_str] # Lowercase for case-insensitive matching
    min_choice, max_choice = min(letters), max(letters)

    # Match everything except . and first extract everything after `Answer` tag
    # `<A single letter option` is included since sometimes, 
    # the model repeats the question during inference and the parser should not 
    # consider that
    pattern_answer = r"\**Answer\**\s*:\**\s*(?!<A single letter option)([^.\s$]+)"
    match = re.search(pattern_answer, json_string, re.IGNORECASE)

    if match:
        answer_string = match.group(1)
        answer_string = re.sub(r'[^\w\s]|[\s]+$', '', answer_string)  # remove all trailing space and punctuation
    else:
        answer_string = None


    # Strict check:
    # Models sometimes directly output `Answer: Southwest.` 
    # instead of a choice letter.
    # For such cases, we expect the model to predict the correct choice string
    # if it's only going predict the choice string. So, rephrased answers will
    # be tagged as non direct answer and parsing proceeds to else block
    is_direct_answer = None
    if answer_string is not None:
        is_direct_answer = choice_str_lower.index(answer_string.lower()) if answer_string.lower() in choice_str_lower else None

    if is_direct_answer is not None:
        # This occurs when the model predicts the correct choice string
        # Map back to the corresponding choice letter
        answer_letter = letters[is_direct_answer]
        answer_text = answer_string
    else:
        # Ignore the phrase from question, some models repeat question
        # First check if there's an "Answer:" tag in the response
        search_answer = re.search(r"\**Answer\**\s*:\**\s*(?!.*<A single letter option that best answers the question>).*", json_string, re.IGNORECASE)
        if search_answer is None:
            logging.info(f"No Answer tag found, match option directly")
            optional_phrase = "the correct answer is"
            # Find the pattern with choice letter and optional preceeding phrase string
            # Match until \n or . or Reason is encountered
            pattern = fr'(?:\b{optional_phrase}\b)*\s*\**([{min_choice}-{max_choice}])\s*\.?\s*(.*?)(?=\**Reason\**\b|\n|\.|$)\.?'
        else:
            # "Answer:" tag found, extract the choice letter after it
            # Stop extraction when encountering "Reason:", newline, or period
            pattern = fr'\**Answer\**\s*:\**\s*([{min_choice}-{max_choice}])\**\s*\.?\s*(.*?)(?=\**Reason\**\b|\n|\.|$)\.?'  
        match = re.search(pattern, json_string, re.IGNORECASE | re.DOTALL)
        if match:
            answer_letter = match.group(1)  # The choice letter (A, B, C, etc.)
            answer_text = match.group(2)    # Any additional text after the letter
            logging.info(f"Found answer: {answer_letter} {answer_text}")
        else:
            logging.info("No match found for non direct answer")
            answer_letter = answer_text = None
    
    return answer_letter, answer_text, is_direct_answer

def extract_reason(json_string):
    """
    Extract the reasoning text from the model's response.
    
    Args:
        json_string (str): The model's complete response text
    
    Returns:
        str or None: The extracted reasoning text, or None if no reasoning found
    """
    # Pattern to match "Reason:" followed by the reasoning text
    pattern = r"\**Reason\**\s*:*\**\s*(.*)\n?"
    match = re.search(pattern, json_string, re.IGNORECASE)
    return match.group(0) if match else None
    
    

def evaluate_model_accuracy(model_output_path: str, eval_summary_path: str, model_name: Optional[str] = None) -> Tuple[float, int]:
    """
    Evaluate the accuracy of a model based on its output file.
    
    This function processes a JSONL file containing model responses, extracts answers,
    compares them against ground truth, and calculates accuracy metrics.
    
    Args:
        model_output_path (str): Path to the JSONL file containing model outputs
        eval_summary_path (str): Path where evaluation summary will be saved
        model_name (str, optional): Name of the model being evaluated
    
    Returns:
        Tuple[float, int]: (accuracy_score, total_questions_processed)
    """
    eval_summary: List[Dict[str, str]] = []
    correct_answers = 0
    line_count = 0
    
    with open(model_output_path, "r") as f:
        for line in f:
            data = json.loads(line)
            question_id = int(data["id"].split('.')[-1])
            task = data["id"].split(".")[0]
            id_q = data["id"]
            logging.info(f"----- id: {id_q} -------")

            line_count += 1
            try:
                # Extract the available multiple choice options from the prompt
                choices = extract_available_options(data["prompt"])

                # Parse the model's response to extract its answer
                answer_letter, answer_text, is_direct_answer = extract_answer(data["answer"], choices)

                # Extract any reasoning the model provided
                reason = extract_reason(data["answer"])

                # Get the ground truth answer and lower for comparison
                ref_ans = str(data["oracle_option"]).lower()
                logging.info(f"-- ref_answer: {ref_ans}")
                logging.info(f"-- answer_letter: {answer_letter}, answer_text: {answer_text}, is_direct_answer: {is_direct_answer is not None}, is_direct_answer_idx: {is_direct_answer}")
                logging.info(f"-- total_response: {data['answer']}")
                logging.info(f"-- choices: {choices}")

                # Compare model answer with ground truth
                model_answer = str(answer_letter).lower()
                if model_answer is not None:
                    # Note: only considers option, not the text after the choice
                    eval_result = int(ref_ans == model_answer) 
                else:
                    eval_result = 0
                    logging.info("--- No answer choice was extracted")

                correct_answers += eval_result
                eval_summary.append({
                    "id": id_q,
                    "ref": ref_ans, 
                    "model_output": model_answer, 
                    "eval_result": eval_result, 
                    "is_direct_answer": is_direct_answer is not None,
                    "choices": choices,
                    "len_choices": len(choices),
                    "oracle_full_answer": data["oracle_full_answer"], # Complete ground truth answer
                    "oracle_answer": data["oracle_answer"], # Ground truth answer text
                    "model_answer_text_extracted": answer_text, # Text extracted after "Answer:"
                    "model_reason_text_extracted": reason, # Text extracted after "Reason:"
                    "model_complete_response": data["answer"]  # Full model response
                    
                    })
                logging.info(f"-----------------------\n\n")

            except Exception as e:
                print(e)
                continue
    
    # Save evaluation results to CSV file
    with open(eval_summary_path.replace(".jsonl", ".csv"), mode='w') as csv_file:
        fieldnames = list(eval_summary[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_summary)

    return correct_answers / line_count if line_count > 0 else 0, line_count


def main(args):
    output_dir = os.path.join(args.output_dir, args.task)
    output_csv = os.path.join(args.eval_summary_dir, f"{args.task}_acc.csv")

    model_accuracies = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".jsonl"):
            model_name = extract_model_name(filename)
            if model_name:
                output_filename = os.path.join(output_dir, filename)
                eval_summary_path = os.path.join(args.eval_summary_dir, f"{args.task}_{model_name}_eval_summary.jsonl")
                accuracy, num_outputs = evaluate_model_accuracy(output_filename, eval_summary_path, model_name)
                model_accuracies.append({'Model Name': model_name, f'Acc': accuracy})
                logging.info(f"Parsed answers at {eval_summary_path}")
                logging.info(f"-------- {args.task}: ACCURACY: {accuracy} -------")

    df = pd.DataFrame(model_accuracies)
    df_sorted = df.sort_values(by='Model Name', ascending=True)
    df_sorted.to_csv(output_csv, index=False)
    logging.info(f"{args.task} | {args.mode} | CSV file with model accuracies has been created at {output_csv}")
    


if __name__ == '__main__':
    args = parse_arguments()
    args.output_folder = os.path.join(args.output_folder, args.dataset_id.replace("/", "__"))
    args.output_dir = os.path.join(args.output_folder, args.mode)
    args.eval_summary_dir = os.path.join(args.eval_summary_dir, args.mode)
    os.makedirs(args.eval_summary_dir, exist_ok=True)

    main(args)
