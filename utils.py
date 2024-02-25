import os 
import pandas as pd
import ast

def directory_creator(directory: str, new_subdir: str) -> str:
    """
    Creates a new subdirectory within the specified directory. If the specified directory is empty,
    it uses the current working directory. The function checks if the new subdirectory exists before
    creation to avoid duplication.

    Args:
        directory (str): The parent directory in which the new subdirectory will be created. If empty,
                         the current working directory is used.
        new_subdir (str): The name of the new subdirectory to create.

    Returns:
        str: The path to the newly created subdirectory.
    """
    
    # Use current working directory if directory is empty
    if not directory:
        directory = os.getcwd()
    new_directory = os.path.join(directory, new_subdir)
    
    # Create the new subdirectory if it does not exist
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        
    return new_directory

def load_df_and_convert_literals(csv_path: str, 
                                 literal_columns: list[str] = ["user_genres_alltime", 
                                                               "user_genres_morning",
                                                               "user_genres_evening",
                                                               "user_genres_recent",
                                                               'user_top_artists',
                                                               "song_history", 
                                                               "recommended_song"]
                                 ) -> pd.DataFrame:
    """
    Loads a DataFrame from a CSV file and converts specified columns from string representations of Python
    literals (e.g., lists, dictionaries) back into Python objects using ast.literal_eval. This is particularly
    useful for columns that were saved with complex data types.

    Args:
        csv_path (str): The file path to the CSV file to be loaded.
        literal_columns (list[str]): A list of column names in the CSV whose string contents should be
                                     evaluated as Python literals.

    Returns:
        pd.DataFrame: A pandas DataFrame with the specified columns converted back from strings to Python objects.
    """
    # Load the DataFrame from CSV
    df = pd.read_csv(csv_path)
    
    # Convert specified columns using ast.literal_eval
    for col in literal_columns:
        if col in df.columns:
            print(col)
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else x)
    
    return df
