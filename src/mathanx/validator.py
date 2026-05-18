# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pydantic",
#     "torch",
#     "transformers"
# ]
# ///

"""
This module contains the implementation of the DataValidator, an object whose main task is to perform the validation of the correctness of the schemas in the data generated.


Author: Anthony Tricarico
Email: tricarico672@gmail.com
"""

import json
import os
from pathlib import Path
# import sys
from typing import Any, Dict

from pydantic import ValidationError

from mathanx.schemas import Call1Schema, Call2Schema, Call3Schema, Call4Schema
from mathanx.constants import DATA_PATH, MODEL_NAME_MAPPING

# sys.path.append(".")


class DataValidator:
    """
    The DataValidator class serves as a common interface to validate the data generated.
    The class handles:
        1. Reading the JSON files.
        2. Validating their structure and performing other checks.

    Internal methods are prefixed with an underscore (_). Those methods should not be used directly
    by the user outside the class.

    Users of the class should always use public methods.
    """

    def __init__(self, dir_path: str) -> None:
        self.dir_path = Path(dir_path).resolve().absolute()
        self.correct_files: Dict = {"files": {}}
        self.wrong_files: Dict = {"files": {}}
        self._pending_writes: Dict = {}
        self.proportions: Dict = {"files": {}}

        if not self.dir_path.is_dir():
            raise OSError(f"Path {self.dir_path} is not a directory!")

    def _read_json(
        self, file_path: str
    ) -> Call1Schema | Call2Schema | Call3Schema | Call4Schema:
        """
        This function is used internally to read JSON files composing the dataset.

        Args:
            file_path: str The path to the JSON file to read.
        Returns:
            A dictionary containing the parsed JSON from one of the four calls.
        """
        if ".json" not in str(file_path):
            raise ValueError(f"{file_path} is not a valid json file")
        with open(file_path, "r") as f:
            return json.load(f)

    def _write_wrong(
            self,
            model_name: str,
            run_id: str,
            task: str,
            error: Any) -> None:
        """
        This function is used to save the runs that exhibited structural or content inconsistencies.

        Args:
            model_name: str The name of the LLM generating the specific data record.
            run_id: str The unique identifier of the run.
            task: str The name of the task (e.g., "summarization", "translation").
            error: Any The error encountered.
        Returns:
            None since all modifications happen in-place on self.wrong_files.
        """
        # 1. Ensure the model key exists
        if model_name not in self.wrong_files["files"]:
            self.wrong_files["files"][model_name] = {}

        # 2. Ensure the task key exists within that model
        if task not in self.wrong_files["files"][model_name]:
            self.wrong_files["files"][model_name][task] = []

        # 3. Create a structured record for the error
        error_record = {
            "run_id": str(run_id),
            "error": str(error)
        }

        # 4. Append the record to the task list
        self.wrong_files["files"][model_name][task].append(error_record)

    def _write_correct(
            self,
            model_name: str,
            task: str,
            run_id: str) -> None:
        """
        This function is used to save the runs that did not exhibit structural or content inconsistencies.

        Args:
            model_name: str The name of the LLM generating the specific data record.
            run_id: str The unique identifier of the run.
            task: the name of the task
        Returns:
            None since all modifications happen in-place on self.correct_files.
        """
        # 1. Ensure the model key exists
        if model_name not in self.correct_files["files"]:
            self.correct_files["files"][model_name] = {}

        # 2. Ensure the task key exists within that model
        if task not in self.correct_files["files"][model_name]:
            self.correct_files["files"][model_name][task] = []

        # 3. Safe to append now
        self.correct_files["files"][model_name][task].append(str(run_id))

    def _check_call(self, file_path: str, processed_dir: str | None = None) -> None:
        """Validates the payload and saves it to the mirrored processed directory if successful."""

        # 1. Read the raw data
        try:
            data = self._read_json(file_path)
        except Exception as e:
            self._write_wrong("unknown", "unknown",
                              "unknown", f"Read Error: {e}")
            return

        model_name = data.get("model", "unknown_model")  # type: ignore
        task = data.get("task", "unknown_task")  # type: ignore
        run_id = data.get("run_id", "unknown_run_id")  # type: ignore
        call_name = data.get("call_name", "unknown_run_id")  # type: ignore

        try:
            # 2. Validate (This automatically triggers your mode='before' repairs)
            if task == "call1":
                validated_data = Call1Schema.model_validate(data)
            elif task == "call2":
                validated_data = Call2Schema.model_validate(data)
            elif task == "call3":
                validated_data = Call3Schema.model_validate(data)
            elif task == "call4":
                validated_data = Call4Schema.model_validate(data)
            else:
                self._write_wrong(model_name, run_id, task,
                                  f"Unknown Task '{task}'")
                return

            # 3. Queue for Processed Directory mirroring
            if processed_dir:
                rel_path = os.path.relpath(file_path, self.dir_path)
                dest_path = os.path.join(processed_dir, rel_path)

                # Initialize the dictionary structure if needed
                if model_name not in self._pending_writes:
                    self._pending_writes[model_name] = {}
                if run_id not in self._pending_writes[model_name]:
                    self._pending_writes[model_name][run_id] = []

                # Queue the file instead of writing it immediately
                self._pending_writes[model_name][run_id].append({
                    "dest_path": dest_path,
                    "content": validated_data.model_dump_json(indent=2, exclude_none=True, by_alias=True)
                })

        except ValidationError as e:
            formatted_errors = []
            for err in e.errors():
                field = ".".join([str(loc) for loc in err["loc"]])
                if err["type"] == "missing":
                    formatted_errors.append(
                        f"Missing required field: '{field}'")
                else:
                    formatted_errors.append(
                        f"Field '{field}' failed: {err['msg']}")

            error_string = " | ".join(formatted_errors)
            # CHANGE: Use call_name instead of task
            self._write_wrong(model_name=model_name, run_id=run_id,
                              task=call_name, error=error_string)
            return
        except Exception as e:
            # CHANGE: Use call_name instead of task
            self._write_wrong(model_name, run_id, task=call_name,
                              error=f"Unexpected Error: {e}")
            return

        # CHANGE: Use call_name instead of task
        self._write_correct(model_name, call_name, run_id)

    def validate_data(self, processed_dir: str | None = None, external_wrong_path: str | None = None) -> None:
        """Reads raw data, validates it, and processes them preserving structure."""
        if not os.path.exists(self.dir_path):
            print(f"Directory not found: {self.dir_path}")
            return

        model_dirs = [d for d in os.listdir(
            self.dir_path) if os.path.isdir(os.path.join(self.dir_path, d))]

        for model in model_dirs:
            single_model_dir = os.path.join(self.dir_path, model)
            print(f"Processing model: {model}")

            for file in os.listdir(single_model_dir):
                if not file.endswith(".json"):
                    continue

                file_path = os.path.join(single_model_dir, file)
                self._check_call(file_path, processed_dir)

        # FLUSH ONLY 100% CORRECT RUNS TO DISK
        if processed_dir and self._pending_writes:
            print("\nCalculating complete runs and writing to disk...")
            if external_wrong_path:
                valid_run_ids_by_model = self.get_valid_run_ids(
                    external_wrong_path=external_wrong_path)
            else:
                valid_run_ids_by_model = self.get_valid_run_ids()

            total_files_written = 0

            for model, valid_runs in valid_run_ids_by_model.items():
                if model in self._pending_writes:
                    for run_id in valid_runs:
                        # If the run_id is in our valid intersection, write its 5 queued files
                        if run_id in self._pending_writes[model]:
                            for file_data in self._pending_writes[model][run_id]:
                                dest_path = file_data["dest_path"]
                                content = file_data["content"]

                                os.makedirs(os.path.dirname(
                                    dest_path), exist_ok=True)
                                with open(dest_path, "w", encoding="utf-8") as f:
                                    f.write(content)
                                total_files_written += 1

            print(f"Success! Wrote {
                  total_files_written} fully validated files to '{processed_dir}'.")

    # Updated type hint
    def get_number_wrong_files_by_model(self) -> Dict[str, Dict[str, int]]:
        if not self.wrong_files:
            self.validate_data()

        wrong_ids = dict()
        for model in self.wrong_files["files"]:
            wrong_ids[model] = {}

            for call in self.wrong_files["files"][model]:
                wrong_ids[model][call] = len(
                    self.wrong_files["files"][model][call])
        return wrong_ids

    def get_number_correct_files_by_model(self) -> Dict[str, Dict[str, int]]:
        """
        Count the number of correct run_ids for each model and task.
        """
        # ensure data is loaded
        if not self.correct_files:
            self.validate_data()

        correct_counts = dict()

        # Iterate through the models
        for model in self.correct_files["files"]:
            # Initialize the nested dictionary for this model
            # This prevents KeyError.
            correct_counts[model] = {}

            # Iterate through the tasks (calls)
            for task in self.correct_files["files"][model]:
                # Count the number of run_ids in the list
                count = len(self.correct_files["files"][model][task])
                correct_counts[model][task] = count

        return correct_counts

    def get_detail_correct_files_by_model(self) -> Dict[str, int]:
        """
        Get the dictionary containing the detail about the correct files
        """
        if not self.correct_files:
            self.validate_data()

        return {
            model: {call: val for call, val in calls.items()}
            for model, calls in self.correct_files.get("files", {}).items()
        }  # type: ignore

    def get_valid_run_ids(self, external_wrong_path: str | None = None) -> Dict[str, set]:
        """
        Returns a dictionary mapping model names to a set of 100% correct run_ids.
        Optionally accepts a path to external JSON files (or a deep directory of them) 
        containing wrong run_ids to exclude.
        """
        # Load the external blocklist into a single giant Set
        external_bad_call1_ids = set()

        if external_wrong_path and os.path.exists(external_wrong_path):
            if os.path.isdir(external_wrong_path):
                # os.walk searches the main folder AND all subfolders
                for root, _, files in os.walk(external_wrong_path):
                    for file in files:
                        if file.endswith(".json"):
                            file_path = os.path.join(root, file)
                            self._extract_bad_ids(
                                file_path, external_bad_call1_ids)

            elif os.path.isfile(external_wrong_path) and external_wrong_path.endswith(".json"):
                self._extract_bad_ids(
                    external_wrong_path, external_bad_call1_ids)

        res = dict()
        for model in self.correct_files.get("files", {}):
            model_data = self.correct_files["files"][model]

            # Get Pydantic-approved lists
            correct1 = set(model_data.get("call1_topics", []))

            # APPLY THE BLOCKLIST: Remove any IDs flagged by external file
            if external_bad_call1_ids:
                correct1 = correct1 - external_bad_call1_ids

            correct2 = set(model_data.get("call2_scales", []))
            correct3_b1 = set(model_data.get("call3_forma_mentis_batch1", []))
            correct3_b2 = set(model_data.get("call3_forma_mentis_batch2", []))
            correct4 = set(model_data.get("call4_msesr_mcq", []))

            # Calculate final intersection
            overall_correct = correct1.intersection(
                correct2, correct3_b1, correct3_b2, correct4
            )
            res[model] = overall_correct

        return res

    def _extract_bad_ids(self, file_path: str, bad_id_set: set) -> None:
        """Helper to read the colleague's JSON and add the IDs to the blocklist."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Navigates their structure: {"call_1": {"run_id": [...]}}
                wrong_ids = data.get("call_1", {}).get("run_id", [])
                bad_id_set.update(wrong_ids)
        except Exception as e:
            print(f"Error reading external wrong file {file_path}: {e}")

    def get_correct_files(self) -> Dict[str, int]:
        """
        This function computes the total number of valid runs for each model.
        The number of correct runs is operationalized as the runs that did not
        raise any validation error across each of the four calls.
        For speed, this has been implemented as the intersection of the different
        sets of correct ids for each run.

        Args:
            No args required.
        Returns:
            A dictionary having as key the name of the model and as value the
            number of correct files identified for that specific model.
        """

        if not self.correct_files:
            self.validate_data()

        valid_ids = self.get_valid_run_ids()
        return {model: len(runs) for model, runs in valid_ids.items()}

    def get_mode_proportions(self):
        """
        This method is used to check how many calls belong
        to the llm and human mode.

        Returns:
            A dictionary compatible with the JSON standard.
        """

        model_dirs = [d for d in os.listdir(
            self.dir_path) if os.path.isdir(os.path.join(self.dir_path, d))]

        for model in model_dirs:
            # set up dict for each model, keys should be llm, and human
            single_model_dir = os.path.join(self.dir_path, model)
            print(f"Extracting proportions for model: {model}")

            count_llm = 0
            count_human = 0

            for file in os.listdir(single_model_dir):
                # only analyze call1 records to avoid duplicates
                if not file.endswith(".json") or not "call1" in file:
                    continue

                file_path = os.path.join(single_model_dir, file)
                data = self._read_json(file_path=file_path)

                if data["mode"].strip() == "llm":  # type: ignore
                    count_llm += 1
                elif data["mode"].strip() == "human":  # type: ignore
                    count_human += 1

            self.proportions["files"][model] = {
                "llm": count_llm, "human": count_human}


if __name__ == "__main__":
    processed_dir = Path(DATA_PATH).resolve().absolute()
    dv = DataValidator(processed_dir)  # type: ignore
    dv.validate_data()

    count_correct: Dict = dv.get_correct_files()

    reviewed_names = dict()

    for k in count_correct:
        new_name = MODEL_NAME_MAPPING[k]
        if new_name not in reviewed_names:
            reviewed_names[new_name] = count_correct[k]
        else:
            reviewed_names[new_name] += count_correct[k]

    print(reviewed_names)
