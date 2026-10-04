"""
Smart Study Notes Generator
===========================
Aim:
Develop a Python application that accepts a paragraph of text as input and uses
a pre-trained Generative AI model (Hugging Face Transformers) to:
1. Generate a concise, coherent summary of the paragraph.
2. Extract a structured set of key points (study takeaways).
3. Display original word count and summary word count.
4. Calculate the percentage by which the text was reduced.

Pre-trained Model: Falconsai/text_summarization (T5-based Seq2Seq architecture)
"""

import os
import re
import sys
import time
from typing import Dict, List, Any, Optional

# Disable unnecessary warnings from huggingface
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install them with: pip install torch transformers sentencepiece")
    sys.exit(1)


class SmartStudyNotesGenerator:
    """
    Generator class that encapsulates the pre-trained Generative AI model
    for text summarization, study point extraction, and reduction analytics.
    """

    DEFAULT_MODEL_NAME = "Falconsai/text_summarization"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        """
        Initialize the model and tokenizer.
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[*] Initializing Smart Study Notes Generator on [{self.device.upper()}]...")
        start_t = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        load_time = round(time.time() - start_t, 2)
        print(f"[+] Model '{self.model_name}' ready in {load_time}s.")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespaces and remove unexpected artifacts."""
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    @staticmethod
    def _format_sentence_endings(text: str) -> str:
        """Ensure the summary completes cleanly at a sentence boundary and starts capitalized."""
        text = text.strip()
        if not text:
            return ""
        # Capitalize first letter
        text = text[0].upper() + text[1:]
        # Locate the last terminal punctuation
        last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_punct != -1:
            return text[:last_punct + 1]
        return text + '.'

    def _extract_key_concepts(self, text: str) -> List[str]:
        """
        Extract high-salience terms/concepts from the input paragraph
        to assist students with key vocabulary.
        """
        # Filter common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'while', 'because', 'as',
            'until', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'can', 'will', 'just', 'should', 'now', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'these', 'those',
            'this', 'that', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
            'also', 'often', 'such', 'well', 'many', 'much'
        }
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]{3,}\b', text)
        freq = {}
        for w in words:
            wl = w.lower()
            if wl not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        
        # Sort by frequency and length
        sorted_terms = sorted(freq.keys(), key=lambda k: (freq[k], len(k)), reverse=True)
        # Deduplicate case-insensitively
        seen = set()
        unique_terms = []
        for term in sorted_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
            if len(unique_terms) >= 6:
                break
        return unique_terms

    def _extract_key_points(self, text: str, summary: str) -> List[str]:
        """
        Derive structured, bulleted study takeaways from the AI-generated
        summary and input paragraph.
        """
        # Split summary into distinct sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
        key_points = []

        for sent in raw_sentences:
            s = sent.strip()
            if len(s) > 15:
                # Ensure capitalized first letter
                s = s[0].upper() + s[1:]
                if not s.endswith(('.', '!', '?')):
                    s += '.'
                key_points.append(s)

        # If summary produced fewer than 2 sentences, supplement with key clauses
        if len(key_points) < 2:
            clauses = re.split(r'[,;]\s+', text)
            for c in clauses:
                c_clean = c.strip()
                if 25 < len(c_clean) < 120 and c_clean not in summary:
                    c_clean = c_clean[0].upper() + c_clean[1:]
                    if not c_clean.endswith('.'):
                        c_clean += '.'
                    key_points.append(c_clean)
                    if len(key_points) >= 3:
                        break

        return key_points

    def _summarize_single_chunk(self, chunk: str, max_tokens: int, min_tokens: int) -> str:
        """Summarize a single text chunk using the T5 model."""
        prompt = "summarize: " + chunk
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_length=max_tokens,
                min_length=min_tokens,
                num_beams=4,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    def generate_notes(
        self,
        paragraph: str,
        target_summary_ratio: float = 0.40,
        min_words: int = 15,
        max_words: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processes any paragraph/document and returns summary, key points,
        word count statistics, and the percentage reduction.
        Handles short text, standard paragraphs, and long multi-paragraph texts.
        """
        cleaned_input = self._clean_text(paragraph)
        if not cleaned_input:
            raise ValueError("Input paragraph cannot be empty.")

        start_time = time.time()
        orig_words = cleaned_input.split()
        orig_word_count = len(orig_words)

        # Dynamic token bounds based on input length
        if orig_word_count <= 25:
            max_tokens = max(6, int(orig_word_count * 0.75))
            min_tokens = max(3, int(orig_word_count * 0.3))
        elif max_words is not None:
            max_tokens = max(15, max_words)
            min_tokens = max(8, min(min_words, max_tokens - 5))
        else:
            calculated_max = int(orig_word_count * target_summary_ratio)
            max_tokens = max(25, min(calculated_max + 15, 140))
            min_tokens = max(10, min(min_words, max_tokens - 10))

        # Check if text is long (> 350 words) requiring multi-chunk summarization
        if orig_word_count > 350:
            words = orig_words
            chunks = []
            chunk_size = 280
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i + chunk_size]))

            chunk_summaries = []
            per_chunk_max = max(30, int(max_tokens / len(chunks)) + 15)
            per_chunk_min = max(10, int(min_tokens / len(chunks)))
            for ch in chunks:
                ch_sum = self._summarize_single_chunk(ch, max_tokens=per_chunk_max, min_tokens=per_chunk_min)
                chunk_summaries.append(ch_sum.strip())

            raw_summary = " ".join(chunk_summaries)
        else:
            raw_summary = self._summarize_single_chunk(cleaned_input, max_tokens=max_tokens, min_tokens=min_tokens)

        summary = self._format_sentence_endings(raw_summary)
        summary_words = summary.split()
        summary_word_count = len(summary_words)

        # Calculate reduction metrics
        words_reduced = orig_word_count - summary_word_count
        if orig_word_count > 0:
            percentage_reduction = round((words_reduced / orig_word_count) * 100, 2)
        else:
            percentage_reduction = 0.0

        percentage_reduction = max(0.0, percentage_reduction)

        # Generate structured key points & vocabulary concepts
        key_points = self._extract_key_points(cleaned_input, summary)
        key_concepts = self._extract_key_concepts(cleaned_input)

        elapsed_time = round(time.time() - start_time, 2)

        return {
            "original_text": cleaned_input,
            "summary": summary,
            "key_points": key_points,
            "key_concepts": key_concepts,
            "original_word_count": orig_word_count,
            "summary_word_count": summary_word_count,
            "words_reduced": words_reduced,
            "percentage_reduction": percentage_reduction,
            "latency_seconds": elapsed_time,
            "model_used": self.model_name
        }


def print_formatted_results(result: Dict[str, Any], title: str = "Smart Study Notes"):
    """Render structured results cleanly in terminal."""
    border = "=" * 76
    sub_border = "-" * 76

    print(f"\n{border}")
    print(f"  {title.upper()}")
    print(border)

    print("\n[ORIGINAL TEXT]")
    print(result["original_text"])

    print("\n[AI GENERATED SUMMARY]")
    print(result["summary"])

    print("\n[KEY STUDY POINTS]")
    for i, pt in enumerate(result["key_points"], 1):
        print(f"  • {pt}")

    if result.get("key_concepts"):
        print("\n[KEY VOCABULARY / CONCEPTS]")
        print("  " + " | ".join(result["key_concepts"]))

    print(f"\n{sub_border}")
    print("  QUANTITATIVE METRICS & WORD REDUCTION")
    print(sub_border)
    print(f"  • Original Word Count  : {result['original_word_count']} words")
    print(f"  • Summary Word Count   : {result['summary_word_count']} words")
    print(f"  • Words Reduced        : {result['words_reduced']} words")
    print(f"  • Percentage Reduction : {result['percentage_reduction']}%")
    print(f"  • Generation Latency   : {result['latency_seconds']} seconds")
    print(f"  • Underlying Model     : {result['model_used']}")
    print(border + "\n")


def interactive_mode():
    """Interactive CLI interface for user inputs."""
    print("=" * 76)
    print("        SMART STUDY NOTES GENERATOR - INTERACTIVE CONSOLE")
    print("=" * 76)
    print("Type or paste your study paragraph below.")
    print("Type 'END' on a new line when finished, or 'QUIT' to exit.\n")

    generator = SmartStudyNotesGenerator()

    while True:
        print("\nEnter your paragraph:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip().upper() == "END":
                break
            if line.strip().upper() == "QUIT":
                print("Exiting application. Goodbye!")
                return
            lines.append(line)

        content = " ".join(lines).strip()
        if not content:
            print("[!] Empty input received. Please provide some text or 'QUIT' to exit.")
            continue

        print("\nGenerating smart summary and study points...")
        try:
            results = generator.generate_notes(content)
            print_formatted_results(results, title="Study Notes Output")
        except Exception as err:
            print(f"[!] Error generating notes: {err}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        # If run directly without args, prompt user or show usage
        print("Running Smart Study Notes Generator...")
        interactive_mode()
