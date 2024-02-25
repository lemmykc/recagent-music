import pandas as pd
import numpy as np
from collections import Counter
from bs4 import BeautifulSoup, Comment
from urllib.request import urlopen
from song import Song
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recagent import RecAgent

class Data:
    """
    TLDR: All loading and querying of data happens in this class. 

    Handles loading, processing, and querying of data for a music recommendation simulation. This class is responsible
    for loading user, song catalog, and streaming data from CSV files, preprocessing this data according to simulation
    requirements, and providing utility functions for accessing specific data slices. It's main purpose is to 
    cross-query between the three linked dataframes. 

    Attributes:
        catalog_path (str): File path for the song catalog data.
        streams_path (str): File path for the user streaming data.
        users_path (str): File path for the user demographic data.
        skip_tolerance (float): The minimum number of seconds a song must be played to not be considered a skip.
        df_catalog (pd.DataFrame): DataFrame containing the song catalog data.
        df_streams (pd.DataFrame): DataFrame containing the user streaming data.
        df_users (pd.DataFrame): DataFrame containing the user demographic data.
        user_ids (List[int]): List of unique user IDs present in the data.
        average_year_of_birth (int): Average birth year of users, for users with a valid year of birth.

    Args:
        catalog_path (str): Path to the CSV file containing the song catalog.
        streams_path (str): Path to the CSV file containing streaming records.
        users_path (str): Path to the CSV file containing user demographics.
        min_stream_history (int): Minimum number of streams a user must have to be included in the data, equal to 
                                length_history_sequence + num_recommendations from the Simulation.
        skip_tolerance (float): Number of seconds defining a skip.
        only_smartradio (bool): Whether to filter streams to only include those from smart radio recommendations, 
                                which indicates there has been a recommendation made by Deezer. 
    """
    def __init__(self,
                 catalog_path: str = 'simcares_data/simcares_20FEB2024_catalog.csv',
                 streams_path: str = 'simcares_data/simcares_20FEB2024_streams.csv',
                 users_path: str = 'simcares_data/simcares_20FEB2024_users.csv',
                 min_stream_history: int = 10,
                 skip_tolerance: float = 59.5,
                 only_smartradio: bool = False
                 ):
        self.catalog_path = catalog_path
        self.streams_path = streams_path
        self.users_path = users_path

        # The skip_tolerance is the number of seconds listened to before a song is considered to be played.
        self.skip_tolerance = skip_tolerance

        # Load data
        self.df_catalog, self.df_streams, self.df_users = self.load_data(min_stream_history, only_smartradio)

        # Get all unique user id's. Note that all user_id's in df_streams are also in df_users.
        self.user_ids = sorted(self.df_users['hashed_user_id'].unique())

        # Calculate average year of df_users year_of_birth if the year is > 1900
        self.average_year_of_birth = self.calculate_average_year_of_birth()


    ### DATA LOADING FUNCTIONS ###

    def load_data(self, min_stream_history: int, only_smartradio: bool) -> tuple:
        """
        Loads and preprocesses the catalog, streams, and users data from CSV files. Filters data based on the provided criteria.

        Args:
            min_stream_history (int): Minimum number of streams required for a user to be included in the analysis.
            only_smartradio (bool): If True, only includes streams that are tagged as coming from smartradio recommendations.

        Returns:
            tuple: A tuple containing three pandas DataFrames: df_catalog, df_streams, df_users.
        """
        # Load the data
        df_catalog = pd.read_csv(self.catalog_path)
        df_streams = pd.read_csv(self.streams_path)
        df_users = pd.read_csv(self.users_path)

        ### STREAMS ###
        # Filter the streams for listen_type=='smartradio' so that they're only recommended ones
        if only_smartradio:
            df_streams = df_streams[df_streams['listen_type'] == 'smartradio']

        # Sort the streams by timestamp
        df_streams = df_streams.sort_values(by=['timestamp'])

        # Filter the streams for users with more than min_stream_history streams
        df_streams = df_streams.groupby('hashed_user_id').filter(lambda x: len(x) > min_stream_history)

        # Convert timestamp cols to pd.Timestamp
        df_streams['timestamp'] = pd.to_datetime(df_streams['timestamp']) # of form "2013-07-26 00:00:00"

        ### USERS ###
        # Filter the users for users with more than min_stream_history streams 
        df_users = df_users[df_users['hashed_user_id'].isin(df_streams['hashed_user_id'].unique())]

        ### CATALOG ###
        # Iterate over release_dates and update the dates to include a default time if missing
        df_catalog['release_date'] = df_catalog['release_date'].apply(lambda x: x if pd.isnull(x) or " " in x else x + " 00:00:00")

        # Now convert the updated strings to datetime objects in catalog
        df_catalog['release_date'] = pd.to_datetime(df_catalog['release_date'], errors='coerce') # of form "2013-07-26"

        return df_catalog, df_streams, df_users
    

    ### QUERYING FUNCTIONS ### 

    def get_random_user(self) -> int:
        """
        Selects a random user ID from the list of users.

        Returns:
            int: A randomly selected user ID.
        """
        rand_user_id = np.random.choice(self.user_ids)
        return rand_user_id

    def calculate_average_year_of_birth(self) -> int:
        """
        Calculates the average year of birth for users who have a valid birth year.

        Returns:
            int: The average year of birth, rounded to the nearest integer.
        """
        filtered_years = self.df_users[self.df_users['year_of_birth'] > 1900]['year_of_birth']
        average_year = int(filtered_years.mean()) # Calculate int for consistency
        return average_year
    
    def get_user_listening_history(self, user_id: int) -> pd.DataFrame:
        """
        Retrieves the full listening history for a specified user from df_streams.

        Args:
            user_id (int): The ID of the user whose listening history is to be retrieved.

        Returns:
            pd.DataFrame: A DataFrame containing the listening history of the specified user.
        """
        df_user_streams = self.df_streams[self.df_streams['hashed_user_id'] == user_id]
        return df_user_streams
    
    def get_user_info_df(self, user_id: int) -> pd.Series:
        """
        Retrieves demographic information for a specified user from df_users.

        Args:
            user_id (int): The ID of the user whose demographic information is to be retrieved.

        Returns:
            pd.Series: A Series containing the demographic information of the specified user.
        """
        user_info_df = self.df_users.loc[self.df_users['hashed_user_id'] == user_id].squeeze()
        return user_info_df
        
    def get_release_year_of_song(self, song_id: int) -> int:
        """
        Retrieves the release year of a specified song.

        Args:
            song_id (int): The ID of the song.

        Returns:
            int: The release year of the song.
        """
        year = self.df_catalog.loc[self.df_catalog['song_id'] == song_id]['release_date'].iloc[0].year
        return year
    
    def get_genres_of_song(self, song_id: int) -> list:
        """
        Retrieves the genres of a specified song.

        Args:
            song_id (int): The ID of the song.

        Returns:
            list: A list of genres associated with the song.
        """
        return self.df_catalog.loc[self.df_catalog['song_id'] == song_id]['genre_tags'].iloc[0]
    
    def get_artists_of_streams(self, df_user_streams: pd.DataFrame) -> pd.Series:
        """
        Retrieves the artists for songs in a user's listening history.

        Args:
            df_user_streams (pd.DataFrame): A DataFrame containing the listening history of a user.

        Returns:
            pd.Series: A Series containing the artists of the songs in the listening history.
        """
        artists = df_user_streams['song_id'].apply(
            lambda x: self.df_catalog.loc[self.df_catalog['song_id'] == x]['artist_name'].iloc[0]
            )
        return artists
    

    ### TRANSFORMATION FUNCTIONS ###

    def transform_user_streams_to_songs(self, 
                                        df_user_streams: pd.DataFrame, 
                                        n_lyric_lines: int) -> list:
        """
        Transforms user stream data into a list of Song instances.

        Args:
            df_user_streams (pd.DataFrame): DataFrame containing the user's streaming data (usually the random subset).
            n_lyric_lines (int): The number of lyric lines to include in the song instance.

        Returns:
            list: A list of Song instances created from the user's streaming data.
        """
        song_history = []
        for _, user_stream in df_user_streams.iterrows():
            song_id = user_stream['song_id']

            # Get the song from the catalog, turn it into a series and then into a Song instance
            catalog_song_df = self.df_catalog.loc[self.df_catalog['song_id'] == song_id].squeeze()
            song = Song(song_id = song_id,
                        name=catalog_song_df['song_title'],
                        artist=catalog_song_df['artist_name'],
                        genres=self.genres_tags_to_list(catalog_song_df['genre_tags']),
                        length=catalog_song_df['song_duration'], # this is in seconds in the data
                        year=self.transform_release_date_to_year(catalog_song_df['release_date']),
                        action=self.transform_listening_time_to_action(user_stream['listening_time']),
                        listen_timestamp=user_stream['timestamp'],
                        n_lyric_lines=n_lyric_lines
                        )
            song_history.append(song)
        return song_history
    
    @staticmethod
    def transform_release_date_to_year(release_date) -> int:
        """
        Converts a release date to just the year. If the release date is missing, returns None.

        Args:
            release_date: The release date of a song.

        Returns:
            int: The year extracted from the release date.
        """
        if release_date is np.nan:
            return None
        if release_date is None:
            return None
        else:
            return release_date.year
    
    def transform_listening_time_to_action(self, listening_time: float) -> str:
        """
        Determines whether a song was played or skipped based on listening time.

        Args:
            listening_time (float): The amount of time a song was listened to.

        Returns:
            str: 'play' if the song was not skipped, 'skip' if it was.
        """
        if listening_time < self.skip_tolerance:
            action = 'play'
        else:
            action = 'skip'
        return action 
    

    ### FORMATTING FUNCTIONS ###
    
    @staticmethod
    def convert_gender(binary_num: int) -> str:
        """
        Converts a binary numerical gender representation into a string.

        Args:
            binary_num (int): The binary representation of gender (0 or 1).

        Returns:
            str: 'male' if binary_num is 0, 'female' if 1.
        """
        gender = 'male' if binary_num == 0 else 'female'
        return gender
    
    @staticmethod
    def genres_tags_to_list(genre_tags: str):
        """
        Converts a pipe-separated string of genre tags into a list.

        Args:
            genre_tags (str): The pipe-separated string of genre tags.

        Returns:
            list: A list of genre tags.
        """
        if genre_tags is np.nan:
            return [None]
        elif type(genre_tags) is not str:
            return [None]
        else:
            return genre_tags.split('|')
    
    
