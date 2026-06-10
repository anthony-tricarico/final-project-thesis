"""
Pydantic v2 schemas for all four experimental calls (Call 1–4) of the MATHANX study.

Author: Anthony Tricarico
Email: tricarico672@gmail.com
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import (
    BaseModel, Field, AliasChoices,
    model_validator, field_validator
)

from mathanx.constants import FORMA_MENTIS_CUES, MAPPING_CALL1_QUESTIONS  # noqa: E402

TOPIC_POOL = list(MAPPING_CALL1_QUESTIONS.keys())


class SemanticAligner:
    def __init__(self, topic_pool: list[str]):
        self.topic_pool = topic_pool
        self._pool_embeddings = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def _lazy_init(self):
        if self._model is not None:
            return
        import torch
        from transformers import BertTokenizer, BertModel

        self._device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._model = BertModel.from_pretrained("bert-base-uncased").to(self._device)
        self._model.eval()

    def _get_embeddings(self, texts: list[str]):
        self._lazy_init()
        tokens = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        import torch
        with torch.no_grad():
            outputs = self._model(**tokens)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def _get_pool_embeddings(self):
        if self._pool_embeddings is None:
            self._pool_embeddings = self._get_embeddings(
                [self.normalize(t) for t in self.topic_pool]
            )
        return self._pool_embeddings

    @staticmethod
    def normalize(text: str) -> str:
        import string
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return " ".join(text.split())

    def align(self, raw_replies: dict[str, str]):
        raw_questions = list(raw_replies.keys())
        if not raw_questions:
            return None, False
        pool_embs = self._get_pool_embeddings()
        raw_embs = self._get_embeddings([self.normalize(q) for q in raw_questions])
        norm_pool = pool_embs / np.linalg.norm(pool_embs, axis=1, keepdims=True)
        norm_raw = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)
        sim_matrix = np.dot(norm_raw, norm_pool.T)

        corrected = {}
        for i in range(len(raw_questions)):
            best_idx = np.argmax(sim_matrix[i])
            if sim_matrix[i][best_idx] >= 0.85:
                corrected[self.topic_pool[best_idx]] = raw_replies[raw_questions[i]]
        return corrected, len(corrected) == len(self.topic_pool)


global_aligner = SemanticAligner(TOPIC_POOL)

# --- BASE CONFIGURATIONS ---
# Percentage of missing associations that could be missing in order to consider
# the file as valid. If more than 25% of the associations are missing then
# the file should be flagged as wrong and dropped.
MISSING_ASSOCIATIONS_PERC: float = 0.25
NUMBER_CUES = 25
REQUIRED_ASSOCIATIONS: int = round(NUMBER_CUES * (1-MISSING_ASSOCIATIONS_PERC))

# --- Nested Models for structured data ---


class OceanDetail(BaseModel):
    score: int
    level: str


class Ocean(BaseModel):
    openness: OceanDetail
    conscientiousness: OceanDetail
    extraversion: OceanDetail
    agreeableness: OceanDetail
    neuroticism: OceanDetail


class ParentsEducation(BaseModel):
    parent_1: Optional[str] = None
    parent_2: Optional[str] = None


class Persona(BaseModel):
    biological_sex: Optional[str] = Field(None, exclude=True)
    gender_identity: Optional[str] = Field(
        None,
        validation_alias=AliasChoices('gender_identity', 'gender'),
        serialization_alias='gender'
    )
    sexual_identity: Optional[str] = Field(
        None,
        validation_alias=AliasChoices('sexual_identity', 'sexual_orientation'),
        serialization_alias='sexual_orientation'
    )

    age: Optional[int] = None
    city_of_living: Optional[str] = None
    employment_status: Optional[str] = None
    education_level: Optional[str] = None
    parents_education: Optional[ParentsEducation] = None
    marital_status: Optional[str] = None
    children: Optional[int] = 0
    migration_status: Optional[str] = None
    religious_beliefs: Optional[str] = None
    hobbies: Optional[List[str]] = []
    fav_subjects: Optional[List[str]] = []
    hat_subjects: Optional[List[str]] = []
    ocean: Optional[Ocean] = None

    @ model_validator(mode='after')
    def clean_and_fix_persona_data(self) -> 'Persona':
        bio = str(self.biological_sex).lower() if self.biological_sex else ""
        gen = str(self.gender_identity).lower() if self.gender_identity else ""

        # Transgender logic correction
        if (bio == "male" and gen == "woman") or (bio == "female" and gen == "man"):
            self.gender_identity = "transgender"

        # Migration status correction
        if self.migration_status:
            m_status = self.migration_status.strip().lower()
            if "native italian" in m_status:
                self.migration_status = "native-born italian"
            elif "native american" in m_status:
                self.migration_status = "native-born american"

        return self


class PersonaStrict(Persona):
    @ model_validator(mode='after')
    def check_consistency(self) -> 'PersonaStrict':
        age = self.age
        children = self.children
        marital = self.marital_status
        edu = self.education_level
        emp = self.employment_status

        if age is None:
            return self

        # OCEAN GENERATION LOGIC
        if self.ocean:
            for trait, data in self.ocean.model_dump().items():
                score, level = data.get('score'), data.get('level')
                if score is not None and level is not None:
                    # Logic directly derived from bucket(score)
                    expected = "low" if score < 33 else (
                        "high" if score > 66 else "moderate")
                    if level != expected:
                        raise ValueError(f"OCEAN {trait}: score {
                                         score} incompatible with level {level}")

        # EDUCATION LOGIC
        if edu:
            valid_edu = []
            if age == 18:
                valid_edu = ["lower secondary",
                             "upper secondary", "vocational diploma"]
            elif age <= 21:
                valid_edu = ["lower secondary", "upper secondary",
                             "vocational diploma", "bachelor's degree"]
            elif age <= 26:
                valid_edu = ["lower secondary", "upper secondary",
                             "vocational diploma", "bachelor's degree", "master's degree"]
            else:
                valid_edu = ["no formal education", "primary school", "lower secondary",
                             "upper secondary", "vocational diploma", "bachelor's degree", "master's degree", "PhD"]

            if edu not in valid_edu:
                raise ValueError(f"Education '{edu}' is impossible for age {
                                 age} based on generation limits.")

        # EMPLOYMENT LOGIC
        if emp and edu:
            valid_emp = []
            if age < 22:
                if edu == "lower secondary":
                    valid_emp = ["employed part-time", "unemployed",
                                 "part-time student", "employed full-time"]
                else:
                    valid_emp = [
                        "full-time student", "employed part-time", "unemployed", "part-time student"]
            elif age < 30:
                if edu in ["bachelor's degree", "master's degree", "PhD"]:
                    valid_emp = ["full-time student", "employed full-time",
                                 "employed part-time", "self-employed", "unemployed", "part-time student"]
                else:
                    valid_emp = [
                        "employed full-time", "employed part-time", "self-employed", "unemployed"]
            elif age < 40:
                if edu in ["master's degree", "PhD"]:
                    valid_emp = ["employed full-time", "employed part-time", "self-employed",
                                 "full-time student", "unemployed", "part-time student"]
                else:
                    valid_emp = [
                        "employed full-time", "employed part-time", "self-employed", "unemployed"]
            elif age <= 65:
                if edu in ["no formal education", "primary school", "lower secondary", "upper secondary"]:
                    valid_emp = [
                        "employed full-time", "employed part-time", "self-employed", "unemployed"]
                else:
                    valid_emp = ["employed full-time", "employed part-time",
                                 "self-employed", "part-time student", "unemployed"]
            # Covers both <= 75 and the final else (> 75) which share the exact same choices
            else:
                valid_emp = ["retired", "employed part-time",
                             "self-employed", "unemployed"]

            if emp not in valid_emp:
                raise ValueError(f"Employment '{emp}' is impossible for age {
                                 age} and education '{edu}'.")

        # MARITAL STATUS LOGIC
        if marital:
            valid_mar = []
            if age < 25:  # Note: In your code, `< 22` and `< 25` have the exact same choices arrays
                valid_mar = ["single", "in a relationship", "married"]
            elif age < 35:
                valid_mar = ["single", "in a relationship",
                             "married", "divorced"]
            else:
                valid_mar = ["single", "in a relationship",
                             "married", "divorced", "widowed"]

            if marital not in valid_mar:
                raise ValueError(f"Marital status '{
                                 marital}' is impossible for age {age}.")

        # CHILDREN LOGIC
        if children is not None and marital:
            # The maximum generated number in ANY array in choose_children is 2.
            if children not in [0, 1, 2]:
                raise ValueError(
                    f"Number of children ({children}) exceeds maximum generated possibilities.")

            # Specific age constraints based on the arrays in choose_children
            if marital in ["single", "in a relationship", "married"]:
                if age < 22 and children not in [0, 1]:
                    raise ValueError(f"Children count '{children}' impossible for age {
                                     age} with marital status '{marital}'.")
            else:
                # For divorced/widowed, allow [0, 1, 2] for all brackets, so no strict restriction here.
                pass

        return self

#######################################
########## CALL 1 Schemas #############
#######################################


class ParsedContentCall1(BaseModel):
    mode: Literal["llm", "human"]
    replies: Dict[str, str]
    reasoning_summary: Optional[str] = None

    @ model_validator(mode='before')
    @ classmethod
    def align_and_validate_replies(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Extract replies
        # Note: Depending on the incoming JSON structure, 'replies' might be nested under 'parsed'
        replies = data.get('replies', {})

        if replies:
            # Run the BERT semantic alignment
            corrected_replies, is_aligned = global_aligner.align(replies)

            if not is_aligned:
                raise ValueError(
                    "ALIGNMENT_ERROR: Questions provided do not sufficiently match the expected TOPIC_POOL.")

            # Replace the raw, unaligned questions with the corrected ones from the pool
            data['replies'] = corrected_replies

        return data


class ResponseParsedCall1(BaseModel):
    task: str
    run_id: str
    mode: Literal["llm", "human"]
    persona: Optional[PersonaStrict] = None
    input_topics: List[str]
    raw_output: str
    parsed: ParsedContentCall1

    @ model_validator(mode='after')
    def check_question_answer_count(self) -> 'ResponseParsedCall1':
        num_questions = len(self.input_topics)
        num_answers = len(self.parsed.replies)
        if num_questions != num_answers:
            raise ValueError(
                f"Mismatch in Count: Found {num_questions} questions in 'input_topics', but {
                    num_answers} answers in 'parsed.replies'."
            )
        return self


class Call1Schema(BaseModel):
    run_id: str
    call_name: str
    task: str
    mode: Literal["llm", "human"]
    persona: Optional[PersonaStrict] = None
    model: str
    base_url: str
    response_parsed: ResponseParsedCall1

    @ model_validator(mode='after')
    def check_root_consistency(self) -> 'Call1Schema':
        if self.mode == "llm" and self.persona is not None:
            raise ValueError("Persona must be null when mode is llm")
        if self.mode == "human" and self.persona is None:
            raise ValueError("Persona missing for human mode")

        rp = self.response_parsed
        if rp.run_id != self.run_id:
            raise ValueError("response_parsed.run_id mismatch")
        if rp.task != self.task:
            raise ValueError("response_parsed.task mismatch")
        if rp.mode != self.mode:
            raise ValueError("response_parsed.mode mismatch")
        return self

#######################################
########## CALL 2 Schemas #############
#######################################


class ScaleItem(BaseModel):
    rating: int = Field(
        ge=1, le=5, description="Rating must be between 1 and 5")
    why: str


class ScaleData(BaseModel):
    items: Dict[str, ScaleItem]


class ParsedContentCall2(BaseModel):
    mode: Literal["llm", "human"]
    persona: Optional[Persona] = None
    scales: Dict[str, ScaleData]

    @ model_validator(mode='before')
    @ classmethod
    def pre_process_and_repair_all(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if 'reasoning_summary' in data and 'parsed' in data:
            data['parsed']['reasoning_summary'] = data.pop('reasoning_summary')

        parsed_data = data.get('parsed', data)
        scales = parsed_data.get('scales', {})

        raw_out = data.get('raw_outputs')
        if isinstance(raw_out, str):
            try:
                raw_dict = json.loads(raw_out)
                if 'scales' in raw_dict and not scales:
                    parsed_data['scales'] = raw_dict['scales']
                    scales = parsed_data['scales']
            except json.JSONDecodeError:
                pass

        if isinstance(scales, dict):
            expected_counts = {'maes': 9, 'amas': 9, 'mseaq': 28}
            for scale_name, expected_count in expected_counts.items():
                if scale_name in scales and isinstance(scales[scale_name], dict):
                    items_dict = scales[scale_name].get('items')

                    if isinstance(items_dict, dict):
                        cleaned_items = {}
                        for k, v in items_dict.items():
                            clean_k = str(k).lower().replace(
                                "items_", "").replace("item_", "").strip()
                            if isinstance(v, dict) and 'rating' in v:
                                try:
                                    rating_val = int(v['rating'])
                                    if rating_val < 1:
                                        v['rating'] = 1
                                    elif rating_val > 5:
                                        v['rating'] = 5
                                except (ValueError, TypeError):
                                    pass
                            cleaned_items[clean_k] = v

                        if len(cleaned_items) > expected_count:
                            trimmed_keys = list(cleaned_items.keys())[
                                :expected_count]
                            cleaned_items = {
                                k: cleaned_items[k] for k in trimmed_keys}

                        scales[scale_name]['items'] = cleaned_items

        return data

    @ model_validator(mode='after')
    def check_scales_completeness(self) -> 'ParsedContentCall2':
        EXPECTED_SCALES = {'maes': 9, 'amas': 9, 'mseaq': 28}
        actual_keys = set(self.scales.keys())
        expected_keys = set(EXPECTED_SCALES.keys())

        if actual_keys != expected_keys:
            raise ValueError(f"Scale Mismatch: Expected keys {
                             expected_keys}, but found {actual_keys}")

        for scale_name, expected_count in EXPECTED_SCALES.items():
            scale_data = self.scales[scale_name]
            actual_count = len(scale_data.items)
            if actual_count != expected_count:
                raise ValueError(f"Incomplete Scale '{scale_name}': Expected exactly {
                                 expected_count} items, but found {actual_count}.")
        return self


class ResponseParsedCall2(BaseModel):
    task: str
    run_id: str
    mode: Literal["llm", "human"]
    persona: Optional[Persona] = None

    # We use a default_factory so it doesn't crash if it's completely missing before the validator catches it
    raw_outputs: Dict[str, str] = Field(default_factory=dict)

    # Apply the alias choice just like we did in Call 3
    parsed: ParsedContentCall2 = Field(
        validation_alias=AliasChoices('parsed', 'model_parsed'),
        serialization_alias='parsed'
    )

    @ model_validator(mode='before')
    @ classmethod
    def pre_process_and_repair(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Un-nest the anomalous 'response_parsed' dictionary
        if 'response_parsed' in data and isinstance(data['response_parsed'], dict):
            nested = data['response_parsed']
            if 'raw_outputs' in nested and 'raw_outputs' not in data:
                data['raw_outputs'] = nested['raw_outputs']
            if 'parsed' in nested and 'parsed' not in data:
                data['parsed'] = nested['parsed']
            if 'model_parsed' in nested and 'model_parsed' not in data:
                data['model_parsed'] = nested['model_parsed']

        # Handle 'model_parsed' alias manually in case the alias decorator needs help
        if 'model_parsed' in data and 'parsed' not in data:
            data['parsed'] = data.pop('model_parsed')

        # Prevent 'missing mode' errors in the nested parsed block
        if 'parsed' in data and isinstance(data['parsed'], dict):
            if 'mode' not in data['parsed']:
                data['parsed']['mode'] = data.get('mode', 'llm')

        # Guarantee parsed exists so the inner scale validations can attempt to run
        if 'parsed' not in data:
            data['parsed'] = {'mode': data.get('mode', 'llm')}

        return data


class Call2Schema(BaseModel):
    run_id: str
    call_name: str
    task: str
    mode: Literal["llm", "human"]
    persona: Optional[Persona] = None
    model: str
    base_url: str
    response_parsed: ResponseParsedCall2

    @ model_validator(mode='after')
    def check_root_consistency(self) -> 'Call2Schema':
        if self.mode == "llm" and self.persona is not None:
            raise ValueError("Persona must be null when mode is llm")
        if self.mode == "human" and self.persona is None:
            raise ValueError("Persona missing for human mode")

        rp = self.response_parsed
        if rp.run_id != self.run_id:
            raise ValueError("response_parsed.run_id mismatch")
        if rp.task != self.task:
            raise ValueError("response_parsed.task mismatch")
        if rp.mode != self.mode:
            raise ValueError("response_parsed.mode mismatch")
        return self

#######################################
########## CALL 3 Schemas #############
#######################################


class FormaMentisItem(BaseModel):
    associations: List[str]
    valence: Dict[str, int]

    @ field_validator('valence')
    @ classmethod
    def check_valence_range(cls, v: Dict[str, int]) -> Dict[str, int]:
        for key, score in v.items():
            if not (1 <= score <= 5):
                raise ValueError(f"Valence score for '{
                                 key}' must be between 1 and 5. Got {score}.")
        return v

    @ field_validator('associations')
    @ classmethod
    def check_associations_constraints(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError(
                f"Expected at least 1 association, found {len(v)}")
        if len(set(v)) != len(v):
            raise ValueError(
                f"Associations must be unique, found duplicates: {v}")
        return v


class ParsedContentCall3(BaseModel):
    mode: Literal["llm", "human"]
    forma_mentis: Dict[str, FormaMentisItem]

    @ model_validator(mode='before')
    @ classmethod
    def clean_forma_mentis(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fm = data.get('forma_mentis', {})
        if isinstance(fm, dict):
            if 'reasoning_summary' in fm:
                fm.pop('reasoning_summary')
        return data

    @ model_validator(mode='after')
    def check_batch_validity(self) -> 'ParsedContentCall3':
        keys = list(self.forma_mentis.keys())
        if len(keys) < REQUIRED_ASSOCIATIONS:
            raise ValueError(f"Forma Mentis size mismatch: Expected at least {
                             REQUIRED_ASSOCIATIONS} associations, found {len(keys)}")

        invalid_words = [
            word for word in keys if word not in FORMA_MENTIS_CUES]
        if invalid_words:
            raise ValueError(
                f"Invalid cue words found (not in FORMA_MENTIS_CUES): {invalid_words}")
        return self

    @ model_validator(mode='after')
    def check_minimal_valence(self) -> 'ParsedContentCall3':
        for cue_word, item_data in self.forma_mentis.items():
            associations_set = set(item_data.associations)
            valence_keys_set = set(item_data.valence.keys())
            scored_associations = associations_set.intersection(
                valence_keys_set)

            if not scored_associations:
                raise ValueError(
                    f"Incomplete Data: For cue word '{
                        cue_word}', none of the generated "
                    f"associations ({
                        item_data.associations}) received a valence score."
                )
        return self


class ResponseParsedCall3(BaseModel):
    task: str
    run_id: str
    mode: Literal["llm", "human"]
    persona: Optional[Persona] = None
    cue_words: List[str]
    parsed: ParsedContentCall3

    @ model_validator(mode='before')
    @ classmethod
    def pre_process_and_repair(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # The "Nested response_parsed" Anomaly (Fixes File 2)
        # Reaches into the mistakenly nested dictionary and hoists the required fields up.
        if 'response_parsed' in data and isinstance(data['response_parsed'], dict):
            nested = data['response_parsed']
            if 'cue_words' in nested and 'cue_words' not in data:
                data['cue_words'] = nested['cue_words']
            if 'parsed' in nested and 'parsed' not in data:
                data['parsed'] = nested['parsed']

        # Handle 'model_parsed' alias (Fixes the issue from your previous question)
        if 'model_parsed' in data and 'parsed' not in data:
            data['parsed'] = data.pop('model_parsed')

        # Fallback: Decode 'raw_output' if 'parsed' is still completely missing
        if 'parsed' not in data and 'raw_output' in data:
            if isinstance(data['raw_output'], str):
                try:
                    raw_dict = json.loads(data['raw_output'])
                    if 'forma_mentis' in raw_dict:
                        data['parsed'] = raw_dict
                except json.JSONDecodeError:
                    pass

        # Fallback: Reconstruct 'cue_words' if the script forgot to generate them
        if 'cue_words' not in data and 'parsed' in data:
            parsed_dict = data['parsed']
            if isinstance(parsed_dict, dict) and 'forma_mentis' in parsed_dict:
                data['cue_words'] = list(parsed_dict['forma_mentis'].keys())

        return data


class Call3Schema(BaseModel):
    run_id: str
    call_name: str
    task: str
    mode: Literal["llm", "human"]
    persona: Optional[Persona] = None
    model: str
    base_url: str
    response_parsed: ResponseParsedCall3

    @ model_validator(mode='after')
    def check_root_consistency(self) -> 'Call3Schema':
        if self.mode == "llm" and self.persona is not None:
            raise ValueError("Persona must be null when mode is llm")
        if self.mode == "human" and self.persona is None:
            raise ValueError("Persona missing for human mode")

        rp = self.response_parsed
        if rp.run_id != self.run_id:
            raise ValueError("response_parsed.run_id mismatch")
        if rp.task != self.task:
            raise ValueError("response_parsed.task mismatch")
        if rp.mode != self.mode:
            raise ValueError("response_parsed.mode mismatch")
        return self

#######################################
########## CALL 4 Schemas #############
#######################################


class ProblemSolvingItem(BaseModel):
    chosen_option: Literal["A", "B", "C", "D", "E"]
    reasoning: str
    confidence_score: int = Field(ge=1, le=5)


class ParsedContentCall4(BaseModel):
    mode: Literal["llm", "human"]
    msesr_problem_solving: Dict[str, ProblemSolvingItem]
    reasoning_summary: str

    @ model_validator(mode='after')
    def check_question_count(self) -> 'ParsedContentCall4':
        expected_count = 18
        actual_count = len(self.msesr_problem_solving)
        if actual_count != expected_count:
            raise ValueError(
                f"Count Mismatch: Expected {
                    expected_count} questions in 'msesr_problem_solving', but found {actual_count}."
            )
        return self


class ResponseParsedCall4(BaseModel):
    task: str
    run_id: str
    mode: Literal["llm", "human"]
    persona: Optional[Persona] = None
    raw_output: str = ""  # Default to empty string to prevent hard crash if missing

    # Apply the alias choice
    parsed: ParsedContentCall4 = Field(
        validation_alias=AliasChoices('parsed', 'model_parsed'),
        serialization_alias='parsed'
    )

    @ model_validator(mode='before')
    @ classmethod
    def pre_process_and_repair(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Un-nest the anomalous 'response_parsed' dictionary
        if 'response_parsed' in data and isinstance(data['response_parsed'], dict):
            nested = data['response_parsed']
            if 'raw_output' in nested and 'raw_output' not in data:
                data['raw_output'] = nested['raw_output']
            if 'parsed' in nested and 'parsed' not in data:
                data['parsed'] = nested['parsed']
            if 'model_parsed' in nested and 'model_parsed' not in data:
                data['model_parsed'] = nested['model_parsed']

        # Handle 'model_parsed' alias manually
        if 'model_parsed' in data and 'parsed' not in data:
            data['parsed'] = data.pop('model_parsed')

        parsed_dict = data.get('parsed', {})
        if not isinstance(parsed_dict, dict):
            parsed_dict = {}

        # FIX FOR CALL 4 ERROR: Propagate 'mode' from parent if missing
        if 'mode' not in parsed_dict:
            parsed_dict['mode'] = data.get('mode', 'llm')

        # Standardize reasoning_summary location
        if 'reasoning_summary' in data:
            parsed_dict['reasoning_summary'] = data.pop('reasoning_summary')

        msesr_problems = parsed_dict.get('msesr_problem_solving', {})
        raw_out = data.get('raw_output')

        if isinstance(raw_out, str):
            try:
                raw_dict = json.loads(raw_out)
                if 'msesr_problem_solving' in raw_dict and not msesr_problems:
                    parsed_dict['msesr_problem_solving'] = raw_dict['msesr_problem_solving']
                    msesr_problems = parsed_dict['msesr_problem_solving']

                if 'reasoning_summary' in raw_dict and 'reasoning_summary' not in parsed_dict:
                    parsed_dict['reasoning_summary'] = raw_dict['reasoning_summary']
            except json.JSONDecodeError:
                pass

        if isinstance(msesr_problems, dict):
            cleaned_problems = {}
            for k, v in msesr_problems.items():
                if not isinstance(v, dict):
                    continue

                nums = re.findall(r'\d+', str(k))
                if not nums:
                    continue
                clean_k = nums[0]

                opt = v.get('chosen_option')
                if isinstance(opt, str):
                    match = re.search(r'[A-Ea-e]', opt)
                    if match:
                        v['chosen_option'] = match.group(0).upper()

                if 'confidence' in v and 'confidence_score' not in v:
                    v['confidence_score'] = v.pop('confidence')

                score = v.get('confidence_score')
                if score is not None:
                    try:
                        score_val = int(score)
                        if score_val < 1:
                            v['confidence_score'] = 1
                        elif score_val > 5:
                            v['confidence_score'] = 5
                    except (ValueError, TypeError):
                        pass

                cleaned_problems[clean_k] = v

            if len(cleaned_problems) > 18:
                keys_to_keep = list(cleaned_problems.keys())[:18]
                cleaned_problems = {
                    k: cleaned_problems[k] for k in keys_to_keep}

            parsed_dict['msesr_problem_solving'] = cleaned_problems

        data['parsed'] = parsed_dict
        return data


class Call4Schema(BaseModel):
    run_id: str
    call_name: str
    task: str
    mode: Literal["llm", "human"]
    persona: Optional[Persona] = None
    model: str
    base_url: str
    response_parsed: ResponseParsedCall4

    @ model_validator(mode='after')
    def check_root_consistency(self) -> 'Call4Schema':
        if self.mode == "llm" and self.persona is not None:
            raise ValueError("Persona must be null when mode is llm")
        if self.mode == "human" and self.persona is None:
            raise ValueError("Persona missing for human mode")

        rp = self.response_parsed
        if rp.run_id != self.run_id:
            raise ValueError("response_parsed.run_id mismatch")
        if rp.task != self.task:
            raise ValueError("response_parsed.task mismatch")
        if rp.mode != self.mode:
            raise ValueError("response_parsed.mode mismatch")
        return self
