import pandas as pd



def filter_survey_results(file_path: str) -> pd.DataFrame:

    """Filters survey results by dropping rows with NaN values in the 'survey_question_1' column,

    renaming columns, and writing the resulting DataFrame to a new CSV file.



    Args:

        file_path: The file path to the CSV file containing the survey results.



    Returns:

        The filtered DataFrame.

    """

    df = pd.read_csv(file_path)

    df = df.dropna(subset=['survey_question_1'])

    df = df.rename(columns={'participant_id': 'participant', 'survey_question_1': 'survey'})

    df.to_csv('survey_results_filtered.csv', index=False)



    return df

