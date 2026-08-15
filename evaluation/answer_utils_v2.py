import re
from typing import Optional, Dict

def check_repetition(text: str) -> bool:
    """Checks if the end of the text is a repetition loop."""
    text = text.strip()
    if len(text) < 150:
        return False
    # Take the last 80 chars and see if they appear earlier in the text
    tail = text[-80:]
    # Check if tail appears before the last 120 chars
    if text.rfind(tail, 0, -120) != -1:
        return True
    
    # Also check if it's repeating short patterns at the end
    tail_30 = text[-30:]
    if text[:-30].endswith(tail_30 * 2):
        return True
        
    return False

def is_extracted_value_placeholder(val: str) -> bool:
    if not val:
        return False
    
    cleaned = val.strip().lower()
    if cleaned.startswith("####"):
        cleaned = cleaned[4:].strip()
        
    if re.fullmatch(r"\[(?:insert\s+)?(?:answer|x|value|\.\.\.| )\]", cleaned, re.IGNORECASE):
        return True
    if cleaned in ["[answer]", "answer", "[insert answer here]", "x", "[]"]:
        return True
    return False

def check_placeholder(text: str) -> bool:
    return False

def parse_options(question: str) -> Dict[str, str]:
    """Extracts MCQ options from the question string.
    Returns a dict mapping letter (e.g. 'A') to option text.
    """
    options = {}
    # E.g. A. 72 or (A) 72
    matches = re.finditer(r"(?:^|\s)(?:\()?([A-Z])(?:\)|\.)\s+(.*?)(?=(?:\s(?:\()?([A-Z])(?:\)|\.)\s+)|$)", question, re.DOTALL)
    
    # Fallback to a simpler split if the complex regex misses
    extracted = re.findall(r"(?:^|\n|\s)(?:\()?([A-Z])(?:\)|\.)\s+([^\n]+)", question)
    for match in extracted:
        letter = match[0].upper()
        content = match[1].strip()
        options[letter] = content
        
    return options

def strip_units(text: str) -> str:
    """Removes common units from a numeric string for pure numeric comparison."""
    # Extract just the first numeric-looking part
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if matches:
        return matches[0]
    return text.strip().lower()

def extract_predicted_answer_v2(prediction: str, is_mcq: bool = False, question: str = "") -> Optional[str]:
    """Extract answer from prediction with uniform rules across conditions."""
    if not prediction or not isinstance(prediction, str):
        return None

    if check_repetition(prediction):
        return None

    extracted = None
    prediction_stripped = prediction.strip()

    if is_mcq:
        upper = prediction_stripped.upper()
        
        # 1. Look for explicit tags
        pats = [
            r"####\s*\[?\(?([A-Z])\)?\]?(?:\.|\b)",
            r"(?:THE\s+)?(?:FINAL\s+)?(?:CORRECT\s+)?ANSWER\s*(?:IS\s*)?[:\-]?\s*\[?\(?([A-Z])\)?\]?(?:\.|\b)",
            r"(?:THE\s+)?(?:CORRECT|RIGHT)\s+(?:CHOICE|OPTION)?\s*(?:IS\s+|:\s*)?\[?\(?([A-Z])\)?\]?(?:\.|\b)",
            r"OPTION\s+\[?\(?([A-Z])\)?\]?(?:\.|\b)\s+IS\s+(?:THE\s+)?CORRECT",
            r"CHOICE\s+\[?\(?([A-Z])\)?\]?(?:\.|\b)\s+IS\s+(?:THE\s+)?CORRECT",
            r"\[(?:ANSWER|CHOICE|OPTION)\s*:?\s*\]?\s*\[?\(?([A-Z])\)?\]?(?:\.|\b)",
            r"THE\s+FINAL\s+ANSWER\s+IS\s+(?:OPTION\s+)?\[?\(?([A-Z])\)?\]?(?:\.|\b)",
            r"(?:STATEMENT|EDIT|ANSWER)\s+IS\s+\(([A-Z])\)",
            r"\[([A-Z])\]",
            r"\\BOXED\{([A-Z])\}",
            r"\(([A-Z])\)\s*$"
        ]
        
        for pat in pats:
            matches = list(re.finditer(pat, upper))
            if matches:
                extracted = matches[-1].group(1)
                break
                
        # 3. Value matching if options are available
        if not extracted:
            val_match = None
            
            # Find the last occurrence of the answer pattern
            matches = list(re.finditer(r"(?:THE\s+)?(?:FINAL\s+)?(?:CORRECT\s+)?ANSWER\s*(?:IS\s*)?[:\-]?\s*(.+)", prediction_stripped, re.IGNORECASE))
            if matches:
                val_match = matches[-1]
            else:
                matches = list(re.finditer(r"####\s*(.+)", prediction_stripped, re.IGNORECASE))
                if matches:
                    val_match = matches[-1]
                    
            if not val_match:
                brackets = re.findall(r"\[(.*?)\]", prediction_stripped)
                if brackets:
                    val_text = brackets[-1].strip()
                    val_match = True
                else:
                    val_match = None
                    
            if val_match and val_match is not True:
                val_text = val_match.group(1).strip()
                
            if val_match:
                letter_start = re.match(r"^\[?\(?([A-Z])\)?\]?[\.\) \:]", val_text, re.IGNORECASE)
                if letter_start:
                    extracted = letter_start.group(1).upper()
                else:
                    val_text_norm = val_text.lower().replace(".", "").strip()
                    val_num = strip_units(val_text)
                    
                    options = parse_options(question) if question else {}
                    if options:
                        for letter, opt_text in options.items():
                            opt_norm = opt_text.lower().replace(".", "").strip()
                            opt_num = strip_units(opt_text)
                            
                            if val_text_norm == opt_norm or val_text_norm.startswith(opt_norm) or opt_norm.startswith(val_text_norm):
                                extracted = letter
                                break
                            if val_num and opt_num and val_num == opt_num:
                                extracted = letter
                                break
                    else:
                        # If no A-J options found in question, assume it expects the raw text
                        extracted = val_text
                        
                    if extracted is None and val_text:
                        extracted = val_text
    else:
        # Numeric extraction (GSM8K)
        text = prediction_stripped
        
        if "####" in text:
            parts = text.split("####")
            last_part = parts[-1].strip()
            matches = re.findall(r"-?\d+(?:\.\d+)?", last_part.replace(",", ""))
            if matches:
                extracted = matches[-1]
                
        if not extracted:
            boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
            if boxed:
                matches = re.findall(r"-?\d+(?:\.\d+)?", boxed[-1].replace(",", ""))
                if matches:
                    extracted = matches[-1]
                
        if not extracted:
            marker_matches = list(re.finditer(r"(?i)(?:answer\s+is|answer:|final\s+answer|so,?\s+we\s+have|equals)\s*([\$€£]?\s*\[?\s*-?\d+(?:,\d{3})*(?:\.\d+)?\s*\]?)", text))
            if marker_matches:
                for m in reversed(marker_matches):
                    val_str = m.group(1)
                    num = re.findall(r"-?\d+(?:\.\d+)?", val_str.replace(",", ""))
                    if num:
                        extracted = num[-1]
                        break

        if not extracted:
            num_matches = list(re.finditer(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text))
            if num_matches:
                for m in reversed(num_matches):
                    end_idx = m.end()
                    trailing = text[end_idx:end_idx+5]
                    if re.search(r"[\+\-\*/=]", trailing.strip()):
                        continue
                    num = re.findall(r"-?\d+(?:\.\d+)?", m.group(0).replace(",", ""))
                    if num:
                        extracted = num[-1]
                        break

    if is_extracted_value_placeholder(extracted) or extracted == "":
        return None
        
    return extracted
