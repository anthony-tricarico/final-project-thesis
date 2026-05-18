"""
This module is a utility that produces a list of sampled
run ids for each model.

There is also the possibility to log and save the extracted
list of run ids as a JSON file, and copy the sampled files 
to a new mirrored directory structure.


Author: Anthony Tricarico
Email: tricarico672@gmail.com
"""

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
from typing import List

import numpy as np


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more verbosity
        format="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger(__name__)

# Schema for the output JSON


@dataclass
class JSONSchema:
    model_name: str
    selected_run_ids: List[str]


class Sampler:
    """
    This is the abstract class defining the interface
    that every sampler should have and a set of related
    methods.
    """

    def __init__(self, path_to_dir, path_to_out_dir=None, path_to_filtered_dir=None) -> None:
        """
        Initialize the object passing and ensuring
        a valid path has been passed.
        """

        self.dir_path = Path(path_to_dir).resolve().absolute()

        if path_to_out_dir:
            self.path_to_out_dir = Path(path_to_out_dir).resolve().absolute()
        else:
            self.path_to_out_dir = None

        if path_to_filtered_dir:
            self.path_to_filtered_dir = Path(
                path_to_filtered_dir).resolve().absolute()
        else:
            self.path_to_filtered_dir = None

        self.sampled_ids: dict[str, list[str]] = {}

        if not self.dir_path.is_dir():
            raise OSError(f"Path {self.dir_path} is not a directory!")

    def _check_level(self) -> bool:
        """
        This helper method determines whether the directory
        passed to the initializer contains other directories
        or contains files.
        """
        lst_files = [f.is_dir() for f in self.dir_path.iterdir()]
        return any(lst_files)

    def _extract_run_id(self, path_to_json: Path):
        """
        This function extracts run_ids from a path
        to a JSON file.
        """
        str_path = str(path_to_json.name)
        if ".json" not in str_path:
            raise ValueError(f"{path_to_json} is not a JSON file.")

        # Find the indices that delimit the run id.
        left_underscore = str_path.find("_", 14)
        right_underscore = str_path.find("_", 16)

        return str_path[left_underscore+1: right_underscore]

    def _write_json(self, file_name: str):
        if self.path_to_out_dir:
            # Create the output directory if it doesn't exist
            self.path_to_out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.path_to_out_dir.joinpath(file_name), "w") as f:
                json.dump(self.sampled_ids, f, indent=2)

    def sample(self, size: int, replacement: bool = False, seed: int = 42, ensure_llm_ratio: bool = False):
        # Initialize the isolated random number generator for reproducibility
        self.rng = np.random.default_rng(seed)

        for model in self.dir_path.iterdir():
            if not model.is_dir():
                continue

            llm_ids = set()
            human_ids = set()

            # only use call1 to extract modes and avoid duplicated ids.
            for file in model.glob("*call1*.json"):
                try:
                    run_id = self._extract_run_id(file)
                except ValueError as e:
                    logger.debug(f"Skipping file extraction error: {e}")
                    continue

                if ensure_llm_ratio:
                    # Read the JSON to check the mode
                    try:
                        with open(file, "r") as f:
                            data = json.load(f)
                        if data.get("mode") == "llm":
                            llm_ids.add(run_id)
                        else:
                            human_ids.add(run_id)
                    except Exception as e:
                        logger.warning(f"Failed to read mode from {
                                       file.name}: {e}. Defaulting to human.")
                        human_ids.add(run_id)
                else:
                    # If flag is off, just dump everything into human_ids to pool them together
                    human_ids.add(run_id)

            llm_list = sorted(list(llm_ids))
            human_list = sorted(list(human_ids))
            all_ids = sorted(list(llm_ids | human_ids))

            if not all_ids:
                logger.warning(
                    f"Directory {model.name} contains no valid JSON files. Skipping.")
                continue

            if ensure_llm_ratio:
                # Calculate the baseline targets (25% LLM, 75% Human)
                target_llm = int(size * 0.25)
                target_human = size - target_llm

                if not replacement:
                    # Case 1: The model doesn't even have enough files total. Take everything.
                    if len(all_ids) < size:
                        logger.warning(f"{model.name} only has {
                                       len(all_ids)} total files (Target: {size}). Taking all available.")
                        actual_llm_target = len(llm_list)
                        actual_human_target = len(human_list)

                    # Case 2: Short on LLM files? Borrow from the human pool to reach total size.
                    elif len(llm_list) < target_llm:
                        shortfall = target_llm - len(llm_list)
                        logger.info(f"{model.name} short on LLM files. Borrowing {
                                    shortfall} from the human pool.")
                        actual_llm_target = len(llm_list)
                        actual_human_target = target_human + shortfall

                    # Case 3: Short on human files? Borrow from the LLM pool to reach total size.
                    elif len(human_list) < target_human:
                        shortfall = target_human - len(human_list)
                        logger.info(f"{model.name} short on human files. Borrowing {
                                    shortfall} from the LLM pool.")
                        actual_human_target = len(human_list)
                        actual_llm_target = target_llm + shortfall

                    # Case 4: Plenty of files in both pools. Stick to the exact 25/75 split.
                    else:
                        actual_llm_target = target_llm
                        actual_human_target = target_human
                else:
                    # Sampling with replacement allows us to blindly hit our exact targets
                    actual_llm_target = target_llm
                    actual_human_target = target_human

                # Sample from both pools using our dynamically adjusted targets
                sampled_llm = self.rng.choice(llm_list, size=actual_llm_target, replace=replacement).tolist(
                ) if actual_llm_target > 0 else []
                sampled_human = self.rng.choice(human_list, size=actual_human_target, replace=replacement).tolist(
                ) if actual_human_target > 0 else []

                sampled_ids = sampled_llm + sampled_human

                # Shuffle so the LLM IDs aren't all clustered at the beginning/end
                self.rng.shuffle(sampled_ids)

            else:
                # Standard unified sampling (Old logic)
                if not replacement and len(all_ids) < size:
                    logger.info(f"{model.name} only has {
                                len(all_ids)} files. Taking all available.")
                    sampled_ids = all_ids
                else:
                    sampled_ids = self.rng.choice(
                        all_ids, size=size, replace=replacement).tolist()

            # Save in dictionary
            if model.name not in self.sampled_ids:
                self.sampled_ids[model.name] = sampled_ids

        if self.path_to_out_dir:
            self._write_json("filtered_run_ids.json")

    def copy_sampled_files(self):
        """
        Mirrors the directory structure and copies the sampled JSON files
        into the specified output directory.
        """
        if not self.path_to_filtered_dir:
            logger.error(
                "No path_to_filtered_dir provided. Cannot copy files.")
            return

        if not self.sampled_ids:
            logger.warning(
                "No sampled IDs found. Did you run .sample() first?")
            return

        logger.info(f"Starting file copy to {self.path_to_filtered_dir}...")

        # Loop through our dictionary of sampled files
        for model_name, run_ids in self.sampled_ids.items():
            run_ids_set = set(run_ids)
            source_model_dir = self.dir_path / model_name
            target_model_dir = self.path_to_filtered_dir / model_name

            target_model_dir.mkdir(parents=True, exist_ok=True)
            copied_count = 0

            # Iterate through all JSON files in the source directory
            for file_path in source_model_dir.glob("*.json"):
                try:
                    current_run_id = self._extract_run_id(file_path)

                    if current_run_id in run_ids_set:
                        destination_path = target_model_dir / file_path.name
                        shutil.copy2(file_path, destination_path)
                        copied_count += 1

                except ValueError as e:
                    logger.debug(f"Skipping file {file_path.name}: {e}")

            logger.info(f"Copied {copied_count} files for model '{
                        model_name}'.")


if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(
        description="Utility to sample and copy run IDs for models.")

    # Directory arguments
    parser.add_argument("--processed_dir", "-p", type=str, required=True,
                        help="Path to the directory containing processed model runs.")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Path to the directory where the filtered_run_ids.json will be saved.")
    parser.add_argument("--filtered_dir", "-f", type=str, default=None,
                        help="Optional: Path to mirror the directory and copy sampled JSON files.")

    # Sampling arguments
    parser.add_argument("--size", "-s", type=int, default=2000,
                        help="Number of files to sample per model. Default is 2000.")
    parser.add_argument("--replacement", "-r", action="store_true",
                        help="Flag to enable sampling with replacement. Default is False.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic sampling. Default is 42.")

    # Ensuring LLM Ratio
    parser.add_argument("--ensure_llm_ratio", "-e", action="store_true",
                        help="Flag to ensure exactly 25% of the sampled dataset is comprised of 'llm' mode files.")

    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve().absolute()
    output_dir = Path(args.output_dir).resolve().absolute()

    filtered_dir = None
    if args.filtered_dir:
        filtered_dir = Path(args.filtered_dir).resolve().absolute()

    logger.info(f"Initializing sampler for {processed_dir}...")
    sampler = Sampler(
        path_to_dir=processed_dir,
        path_to_out_dir=output_dir,
        path_to_filtered_dir=filtered_dir
    )

    sampler.sample(
        size=args.size,
        replacement=args.replacement,
        seed=args.seed,
        ensure_llm_ratio=args.ensure_llm_ratio
    )

    if filtered_dir:
        sampler.copy_sampled_files()

    json_path = output_dir.joinpath("filtered_run_ids.json")
    try:
        with open(json_path, "r") as f:
            json_content = json.load(f)

        print("\n------------- Sampled run ids count summary ------------")
        for model, run_ids in json_content.items():
            print(f"{model}: {len(run_ids)}")

    except FileNotFoundError:
        logger.error(f"\nError: JSON file does not exist at {json_path}!")
