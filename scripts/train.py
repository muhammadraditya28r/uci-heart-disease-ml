from heart_disease.config import RAW_DATA_DIR
from heart_disease.data.ingestion import load_csv
from heart_disease.utils.logging import get_logger



def main() -> None:
    data_path = RAW_DATA_DIR / "heart_disease_uci.csv"

    df = load_csv(data_path)

    print("Dataset loaded succesfully.")
    print(f"Shape: {df.shape}")
    logger = get_logger(__name__)
    logger.info("Loading Dataset")

if __name__ == "__main__":
    main()

