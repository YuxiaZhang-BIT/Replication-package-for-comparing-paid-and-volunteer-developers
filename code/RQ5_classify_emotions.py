import sys
import logging
import argparse
from typing import List, Dict

import pandas as pd
from transformers import pipeline
from tqdm import tqdm


def setup_logging(log_level: int = logging.INFO):
    if not isinstance(log_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify emotions in text data using a transformer model."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing the text data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output CSV file with emotion scores.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="j-hartmann/emotion-english-distilroberta-base",
        help="Hugging Face model name for emotion classification.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing data.")
    return parser.parse_args()


def load_data(input_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(input_path)
        if "content" not in df.columns:
            logging.error("Input CSV must contain a 'content' column.")
            sys.exit(1)
        return df
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        sys.exit(1)


def initialize_classifier(model_name: str):
    try:
        classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device_map="auto",
        )
        return classifier
    except Exception as e:
        logging.error(f"Failed to initialize the classifier: {e}")
        sys.exit(1)


def classify_emotions(classifier, texts: List[str], batch_size: int) -> List[Dict[str, float]]:
    """Classify emotions for a list of texts."""
    try:
        results = classifier(texts, batch_size=batch_size)
        return results
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        sys.exit(1)


def process_results(
    df: pd.DataFrame, classification_results: List[List[Dict[str, float]]]
) -> pd.DataFrame:
    """Process classification results and add them to the DataFrame."""
    emotions = []
    anger_scores = []
    disgust_scores = []
    fear_scores = []
    joy_scores = []
    neutral_scores = []
    surprise_scores = []
    sadness_scores = []

    for res in classification_results:
        score_dict = {item["label"]: round(item["score"], 4) for item in res}
        max_emotion = max(score_dict, key=score_dict.get)
        emotions.append(max_emotion)
        anger_scores.append(score_dict.get("anger", 0))
        disgust_scores.append(score_dict.get("disgust", 0))
        fear_scores.append(score_dict.get("fear", 0))
        joy_scores.append(score_dict.get("joy", 0))
        neutral_scores.append(score_dict.get("neutral", 0))
        surprise_scores.append(score_dict.get("surprise", 0))
        sadness_scores.append(score_dict.get("sadness", 0))

    df["emotion"] = emotions
    df["anger_score"] = anger_scores
    df["disgust_score"] = disgust_scores
    df["fear_score"] = fear_scores
    df["joy_score"] = joy_scores
    df["neutral_score"] = neutral_scores
    df["surprise_score"] = surprise_scores
    df["sadness_score"] = sadness_scores

    return df


def save_results(df: pd.DataFrame, output_path: str):
    """Save the DataFrame to a CSV file."""
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save output file: {e}")
        sys.exit(1)


def main():
    args = parse_arguments()
    setup_logging()

    logging.info("Loading data...")
    df = load_data(args.input)

    logging.info("Initializing classifier...")
    classifier = initialize_classifier(args.model)

    texts = df["content"].astype(str).tolist()
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size

    classification_results = []
    logging.info("Starting emotion classification...")
    for i in tqdm(range(0, len(texts), args.batch_size), total=total_batches, desc="Classifying"):
        batch_texts = texts[i : i + args.batch_size]
        batch_results = classify_emotions(classifier, batch_texts, args.batch_size)
        classification_results.extend(batch_results)

    logging.info("Processing results...")
    df = process_results(df, classification_results)

    logging.info("Saving results...")
    save_results(df, args.output)

    logging.info("Emotion classification completed successfully.")


if __name__ == "__main__":
    main()
