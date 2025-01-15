import os
from datasets import load_dataset
import random
import json
from typing import List, Dict, Any
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

num_categories: int = 16


class WikiCategoryLearner:
    def __init__(
        self,
        output_dir: str = "wiki_categories",
        n_rounds: int = 20,
        samples_per_round: int = 50,
        input_file: str = "enwik8",
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.dataset = load_dataset("LTCB/enwik8")  # For sampling phase
        self.input_file = input_file  # For processing phase
        self.output_dir = output_dir
        self.n_rounds = n_rounds
        self.samples_per_round = samples_per_round
        self.chunk_size = 1000
        self.categories = []
        os.makedirs(output_dir, exist_ok=True)

        # Verify input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_gpt(self, messages: List[Dict[str, Any]]) -> str:
        """Make a GPT API call with retry logic"""
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4 for better category understanding
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in GPT API call: {e}")
            raise

    def get_representative_samples(self) -> str:
        """Get random samples from the dataset"""
        samples = []
        train_size = len(self.dataset["train"])

        for idx in random.sample(range(train_size), self.samples_per_round):
            sample_text = self.dataset["train"][idx]["text"]
            samples.append(sample_text[:1000])

        return "\n---\n".join(samples)

    def _clean_gpt_response(self, response: str) -> str:
        """Clean GPT response by removing markdown code blocks and other formatting"""
        # Remove markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1]
        if "```" in response:
            response = response.split("```")[0]
        return response.strip()

    def get_categories_for_round(self, round_num: int) -> List[Dict]:
        """Get category suggestions for one round of samples"""
        samples = self.get_representative_samples()

        messages = [
            {
                "role": "system",
                "content": "You are an expert in content categorization and taxonomy creation. "
                "Return your response as a raw JSON array without any markdown formatting.",
            },
            {
                "role": "user",
                "content": f"""Round {round_num + 1}/{self.n_rounds}:
                Based on these Wikipedia text samples, suggest exactly {num_categories} broad, distinct categories
                that would be suitable for classifying this Wikipedia content. Categories should be general
                enough to cover most content but specific enough to be meaningful.

                Text samples:
                {samples}

                Respond with exactly {num_categories} categories in this JSON format (no markdown, no code blocks):
                [
                    {{"category": "category_name", "description": "short description"}},
                    ...
                ]""",
            },
        ]

        response = self._call_gpt(messages)
        cleaned_response = self._clean_gpt_response(response)

        try:
            categories = json.loads(cleaned_response)
            if not isinstance(categories, list) or len(categories) != num_categories:
                raise ValueError("Invalid response format")
            return categories
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse GPT response as JSON: {e}\nResponse: {response}"
            )
            raise
        except ValueError as e:
            logger.error(f"Invalid category format: {e}\nResponse: {response}")
            raise

    def aggregate_categories(
        self, all_round_categories: List[List[Dict]]
    ) -> List[Dict]:
        """Synthesize final categories from all rounds"""
        # Flatten all suggestions for the prompt
        all_suggestions = []
        for round_categories in all_round_categories:
            all_suggestions.extend(round_categories)

        messages = [
            {
                "role": "system",
                "content": "You are an expert in content categorization and taxonomy creation. "
                "Synthesize the provided category suggestions into a final set of categories.",
            },
            {
                "role": "user",
                "content": f"""Below are {len(all_suggestions)} category suggestions from multiple rounds
                of Wikipedia content analysis. Synthesize these into exactly {num_categories} distinct, comprehensive
                categories. Consider frequency of similar suggestions and merge related categories.

                Previous suggestions:
                {json.dumps(all_suggestions, indent=2)}

                Provide exactly {num_categories} final categories in valid JSON format like this:
                [
                    {{"category": "category_name", "description": "comprehensive description"}},
                    ...
                ]                
                Ensure categories are broad enough to cover most Wikipedia content but specific
                enough to be meaningful. Avoid overlapping categories.""",
            },
        ]

        response = self._call_gpt(messages)
        cleaned_response = self._clean_gpt_response(response)

        try:
            categories = json.loads(cleaned_response)
            if not isinstance(categories, list) or len(categories) != num_categories:
                raise ValueError("Invalid response format")
            return categories
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse GPT response as JSON: {e}\nResponse: {response}"
            )
            raise

    def classify_chunk(self, text_chunk: str) -> str:
        """Classify a single chunk of text"""
        messages = [
            {
                "role": "system",
                "content": "You are an expert content classifier. "
                "Classify the given text into exactly one category.",
            },
            {
                "role": "user",
                "content": f"""Classify this text chunk into exactly one of these categories:
                {json.dumps([c['category'] for c in self.categories])}

                Text chunk:
                {text_chunk}

                Return only the category name, nothing else.""",
            },
        ]

        response = self._call_gpt(messages)
        response = response.strip()

        # Validate response is one of our categories
        valid_categories = {c["category"] for c in self.categories}
        if response not in valid_categories:
            logger.warning(
                f"Invalid category returned: {response}. Using first category."
            )
            return self.categories[0]["category"]

        return response

    def learn_categories(self) -> List[Dict]:
        """Phase 1: Learn categories through multiple rounds"""
        logger.info(f"Phase 1: Learning categories through {self.n_rounds} rounds...")

        all_round_categories = []

        for round_num in tqdm(range(self.n_rounds), desc="Learning rounds"):
            try:
                round_categories = self.get_categories_for_round(round_num)
                all_round_categories.append(round_categories)

                # Save intermediate results
                with open(
                    os.path.join(self.output_dir, f"categories_round_{round_num}.json"),
                    "w",
                ) as f:
                    json.dump(round_categories, f, indent=2)

            except Exception as e:
                logger.error(f"Error in round {round_num}: {e}")
                continue

        logger.info("Synthesizing final categories...")
        final_categories = self.aggregate_categories(all_round_categories)

        self.categories = final_categories

        with open(os.path.join(self.output_dir, "final_categories.json"), "w") as f:
            json.dump(final_categories, f, indent=2)

        return final_categories

    def process_dataset(self):
        """Phase 2: Process and categorize the dataset from local enwik8 file"""
        logger.info("Phase 2: Categorizing dataset...")

        if not self.categories:
            raise ValueError(
                "Categories not learned yet. Run learn_categories() first."
            )

        # Create subdirectories if they don't exist
        subclass_dir = os.path.join(self.output_dir, "subclasses")
        os.makedirs(subclass_dir, exist_ok=True)

        # Open files for each category
        category_files = {
            category["category"]: open(
                os.path.join(subclass_dir, f"{category['category']}.txt"),
                "w",
                encoding="utf-8",  # Explicitly specify encoding
            )
            for category in self.categories
        }

        chunk_buffer = []
        CHUNK_SIZE = 4000  # Increased chunk size for better context
        BUFFER_SIZE = 5  # Number of chunks to process in one batch

        try:
            with open(self.input_file, "r", encoding="utf-8") as file:
                with tqdm(desc="Processing chunks") as pbar:
                    while True:
                        chunk = file.read(CHUNK_SIZE)
                        if not chunk:
                            break

                        chunk_buffer.append(chunk)
                        pbar.update(len(chunk))

                        # Process buffer when it reaches the desired size
                        if len(chunk_buffer) >= BUFFER_SIZE:
                            self._process_chunk_buffer(chunk_buffer, category_files)
                            chunk_buffer = []

                    # Process any remaining chunks
                    if chunk_buffer:
                        self._process_chunk_buffer(chunk_buffer, category_files)

        finally:
            for f in category_files.values():
                f.close()

    def _process_chunk_buffer(self, chunk_buffer, category_files):
        """Process a buffer of chunks efficiently while preserving exact content"""
        if not chunk_buffer:
            return

        # Join chunks for classification context
        combined_text = "".join(chunk_buffer)

        try:
            # Improved classification prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert content classifier specialized in Wikipedia content. "
                    "Classify the given text into exactly one category based on its main topic.",
                },
                {
                    "role": "user",
                    "content": f"""Classify this Wikipedia text into exactly one of these categories:
                    {json.dumps([c['category'] for c in self.categories])}

                    Consider these classification guidelines:
                    - Focus on the main topic, not minor mentions
                    - Look for key indicators like titles, technical terms, and context
                    - Consider the overall theme rather than individual words
                    
                    Text:
                    {combined_text[:5000]}  # Limit text length for API

                    Return only the category name, nothing else.""",
                },
            ]

            category = self._call_gpt(messages).strip()

            # Validate response
            valid_categories = {c["category"] for c in self.categories}
            if category not in valid_categories:
                logger.warning(
                    f"Invalid category returned: {category}. Using 'Culture and Society' as fallback."
                )
                category = "Culture and Society"

            # Write exact original content, no modifications
            for chunk in chunk_buffer:
                if chunk:  # Write all non-empty chunks exactly as they are
                    category_files[category].write(chunk)

        except Exception as e:
            logger.error(f"Error processing chunk buffer: {e}")
            # Write to fallback category while preserving exact content
            for chunk in chunk_buffer:
                if chunk:
                    category_files["Culture and Society"].write(chunk)


def main():
    learner = WikiCategoryLearner()

    try:
        # Check if categories already exist
        categories_file = os.path.join(learner.output_dir, "final_categories.json")
        if os.path.exists(categories_file):
            # Load existing categories
            with open(categories_file, "r") as f:
                learner.categories = json.load(f)
            logger.info("\nLoaded existing categories:")
            for cat in learner.categories:
                logger.info(f"- {cat['category']}: {cat['description']}")

            # Run only Phase 2
            learner.process_dataset()
            logger.info("\nProcessing complete! Check the wiki_categories directory.")
        else:
            # Run both phases
            logger.info("No existing categories found. Running full process...")
            # Phase 1: Category Learning
            categories = learner.learn_categories()
            logger.info("\nFinal categories:")
            for cat in categories:
                logger.info(f"- {cat['category']}: {cat['description']}")

            # Phase 2: Dataset Processing
            learner.process_dataset()
            logger.info("\nProcessing complete! Check the wiki_categories directory.")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise


if __name__ == "__main__":
    main()
