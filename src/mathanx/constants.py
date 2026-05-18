"""
This script contains a set of constants that are used throughout
the different data processing scripts.

Author: Anthony Tricarico
Email: tricarico672@gmail.com
"""

UNI_EDU_LEVELS = {
    "no formal education", "primary school", "lower secondary", "upper secondary", "vocational diploma",
    "bachelor's degree", "master's degree", "PhD"
}

EMPLOYMENT_STATUS = {
    "employed part-time", "self-employed", "student", "unemployed", "retired"
}

GENDER_LEVELS = {
    "man", "woman", "non-binary", "genderqueer", "agender", "transgender"
}

SEXUAL_ORIENTATION = {
    "heterosexual", "homosexual", "bisexual", "asexual"
}

MARITAL_STATUS = {
    "single", "in a relationship", "married", "divorced", "widowed"
}

MIGRATION_STATUS = {
    "native Italian", "native American", "immigrant"
}

RELIGIOUS_BELIEFS = {
    "Christianity", "Islam", "Hinduism", "Buddhism", "Judaism", "Atheist", "Agnostic"
}

FORMA_MENTIS_CUES = {
    "mathematics", "equation", "numbers", "theorem", "proof",  # math domain knowledge
    # computational thinking
    "informatics", "algorithm", "computation", "problem-solving", "variable",
    "AI", "LLM", "model", "ChatGPT", "data",  # artificial intelligence
    "exam", "grade", "homework", "failure", "success",  # academic assessment
    "class", "lecture", "study", "classroom", "blackboard",  # academic context
    "job", "career", "work", "society", "future",  # work context
    "STEM", "science", "physics", "chemistry", "biology",  # STEM fields
    "art", "music", "literature", "history", "philosophy",  # non-STEM fields
    "creativity", "experiment", "logic", "anxiety", "teamwork",  # skills
    "professor", "teacher", "student", "knowledge", "scientist"  # actors
}

MSESR_CORRECT_ANSWERS = {
    "Q1": "C",
    "Q2": "C",
    "Q3": "D",
    "Q4": "A",
    "Q5": "C",
    "Q6": "D",
    "Q7": "E",
    "Q8": "D",
    "Q9": "C",
    "Q10": "D",
    "Q11": "C",
    "Q12": "D",
    "Q13": "C",
    "Q14": "B",
    "Q15": "C",
    "Q16": "D",
    "Q17": "B",
    "Q18": "E"
}

MAPPING_CALL1_QUESTIONS = {
    "What is your relationship with mathematics?": 1,
    "Do you ever get anxious when thinking about mathematics?": 2,
    "Did you ever use AI to support your math learning in the last year? If yes, how was your experience?": 3,
    "How would you explain, step by step, how to solve a second order algebraic equation?": 4,
    "How would you explain, step by step, how to find the stationary points of an equation y=f(x)?": 5,
    "Briefly, how do you perform a Principal Component Analysis? Should I get anxious about its mathematics? Please, teach me.": 6,
    "According to you, how can LLMs be used to innovate math learning in schools and universities?": 7
}

MODEL_NAME_MAPPING = {
    'deepseek-chat': 'DeepSeek Chat',
    'qwen/qwen3-4b-thinking-2507': 'Qwen3 4B (Thinking)',
    'Qwen/Qwen3-4B-Thinking-2507': 'Qwen3 4B (Thinking)',
    'nvidia/nemotron-3-nano': 'Nemotron-3 Nano',
    'mistralai/ministral-3-3b': 'Ministral 3B',
    'mistral-small-latest': 'Mistral Small 4',
    'ministral-3-3b-reasoning-2512': 'Ministral 3B',
    'mistralai/mistral-small-3.2': 'Mistral Small 3.2',
    'mistral-small-2506': 'Mistral Small 3.2',
    'mistral-small3.2:latest': 'Mistral Small 3.2',  # Unified with the above
    'anita-next-24b-dolphin-mistral-uncensored-ita-i1': 'Anita 24B (Uncensored)',
    'qwen3-4b-instruct-2507-uncensored-unslop-v2': 'Qwen3 4B (Uncensored)',
    'electroglyph/Qwen3-4B-Instruct-2507-uncensored-unslop-v2': 'Qwen3 4B (Uncensored)',
    'ibm/granite-4-h-tiny': 'Granite 4 Tiny',
    'ibm-granite/granite-4.0-h-tiny': 'Granite 4 Tiny',
    'qwen/qwen3.5-9b': 'Qwen3.5 9B',
    'qwen/qwen3.5-9B': 'Qwen3.5 9B',
    'qwen/qwen3-4b-2507': 'Qwen3 4B',
    'Qwen/Qwen3-4B-Instruct-2507': 'Qwen3 4B',
    'mistralai/ministral-3-14b-reasoning': 'Ministral 14B (Reasoning)',
    'ministral-14b-latest': 'Ministral 14B (Reasoning)',
    'microsoft/phi-4-reasoning-plus': 'Phi-4 (Reasoning+)',
    'microsoft/Phi-4-reasoning-plus': 'Phi-4 (Reasoning+)',
    'mistralai/magistral-small-2509': 'Magistral Small',
    'magistral-small-latest': 'Magistral Small',
    'llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b': 'Llama 3.2 MoE 18.4B',
    'grok-4-1-fast-reasoning': 'Grok 4.1 Fast (Reasoning)'
}

FOLDER_NAME_MAPPING = {
    'MANX_LLM_anitamistral': 'Anita 24B (Uncensored)',
    'MANX_LLM_DeepSeekLarge': 'DeepSeek Chat',
    'MANX_LLM_granite4h': 'Granite 4 Tiny',
    'MANX_LLM_Grok41FastReasoning': 'Grok 4.1 Fast (Reasoning)',
    'MANX_LLM_magistralsmall': 'Magistral Small',
    'MANX_LLM_ministral3b': 'Ministral 3B',
    'MANX_LLM_ministral14b': 'Ministral 14B (Reasoning)',
    'MANX_LLM_mistralsmall': 'Mistral Small 3.2',
    'MANX_LLM_MistralSmall4': 'Mistral Small 4',
    'MANX_LLM_phi4reasoning': 'Phi-4 (Reasoning+)',
    'MANX_LLM_qwen4bthink': 'Qwen3 4B (Thinking)',
    'MANX_LLM_qwen4bunce': 'Qwen3 4B (Uncensored)',
    'MANX_LLM_qwen34binstruct': 'Qwen3 4B',
    'MANX_LLM_qwen35_9b': 'Qwen3.5 9B'
}

# Folder paths

DATA_PATH = "data/raw"
