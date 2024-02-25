import pandas as pd
import numpy as np
from typing import TYPE_CHECKING
from data import Data
from recagent import RecAgent
from collections import Counter

if TYPE_CHECKING:
    from data import Data

class User:
    """
    TLDR: Stores and generates metadata about a User and queries their listening history. 

    Represents a user within the music recommendation system, encapsulating their unique characteristics,
    listening history, and preferences. This class interfaces with the Data class to retrieve and process
    user-specific information and generates insights into their music preferences based on their listening history.

    Attributes:
        user_id (int): Unique identifier for the user.
        data (Data): Instance of the Data class providing access to the dataset.
        listening_history (pd.DataFrame): Subset of streaming data specific to the user.
        user_info_df (pd.Series): Demographic information for the user.
        user_traits (UserTraits): Object representing detailed traits and preferences of the user. 

    Args: 
        user_id (int): Unique identifier for the user.
        data (Data): Instance of the Data class providing access to the dataset.
    """

    def __init__(self, user_id: int, data: Data):
        self.user_id = user_id
        self.data = data
        self.listening_history = self.data.get_user_listening_history(user_id) # subset of df_streams
        self.user_info_df = self.data.get_user_info_df(user_id) # subset of df_users
        self.user_traits = self.get_user_traits() # of type UserTraits

    ### RANDOM SUBSETTING ### 
        
    def get_rand_subset_songs_history(self, length_history_sequence: int, 
                                      num_recommendations: int,
                                      n_lyric_lines: int
                                      ) -> tuple[list, list]:
        """
        Selects a random subset of the user's listening history and formats it for recommendation analysis,
        including generating song objects with specified lyric details. 

        Args:
            length_history_sequence (int): The number of songs to include in the history analysis.
            num_recommendations (int): The number of songs to include as recommendations.
            n_lyric_lines (int): The number of lyric lines to include in the song object summaries.

        Returns:
            tuple: Two lists of Song objects representing the user's listening history and recommended songs, respectively.
        """
        random_streams_sequence, recommended_streams = self.get_rand_subset_streams_history(length_history_sequence, num_recommendations)
        song_history = self.data.transform_user_streams_to_songs(random_streams_sequence, n_lyric_lines) # The function needs a dataframe to iterate over rows
        recommended_songs = self.data.transform_user_streams_to_songs(recommended_streams, n_lyric_lines) # The function needs a dataframe to iterate over rows
        
        # Set the age in user_traits according to the (first) recommended stream timestamp
        listen_timestamp_year = recommended_songs[0].listen_timestamp.year
        self.user_traits.update_age(listen_timestamp_year)

        return song_history, recommended_songs

    def get_rand_subset_streams_history(self, length_history_sequence: int, 
                                        num_recommendations: int
                                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Selects a random subset of the user's listening history, dividing it into a sequence for analysis
        and a set of subsequent tracks for recommendation testing.

        Args:
            length_history_sequence (int): The number of tracks to include in the analysis subset.
            num_recommendations (int): The number of tracks to include in the recommendation testing subset.

        Returns:
            tuple: A pair of pd.DataFrame objects representing the selected history and recommendation subsets, 
                    both subsets of listening_history (df_streams).
        """
        # In Data, only users with histories greater than length_history_sequence are listed, so no need for a check here.
        # Calculate the range for possible start indices
        max_start_index = len(self.listening_history) - (length_history_sequence + num_recommendations)

        if max_start_index > 0:
            # Safe to randomly select a start index
            random_start_index = np.random.randint(0, max_start_index)
        elif max_start_index == 0:
            # Only one valid start index, which is 0
            random_start_index = 0

        # The listening history (and "next songs") is size length_history_sequence + num_recommendations, so the start index can be anywhere from 0 to len(listening_history) - (length_history_sequence + num_recommendations).
        #random_start_index = np.random.randint(0, len(self.listening_history) - (length_history_sequence + num_recommendations)) # np.random.randint is exclusive of last value.
        end_history_index = random_start_index + length_history_sequence
        random_streams_sequence = self.listening_history.iloc[random_start_index:end_history_index]
        recommended_streams = self.listening_history.iloc[end_history_index : end_history_index+num_recommendations]
        
        return random_streams_sequence, recommended_streams
    

    ### USER TRAITS ###

    def get_user_traits(self) -> 'UserTraits':
        """
        Retrieves and aggregates various traits and preferences of the user, such as demographic details,
        favorite genres, top artists, and favorite music decade.

        Returns:
            UserTraits: An object encapsulating the user's traits and preferences.
        """
        # Get demographic information
        user_traits = UserTraits(
                    user_id=self.user_id, 
                    gender= self.data.convert_gender(self.user_info_df['gender']),
                    yob= self.user_info_df['year_of_birth'],
                    favorite_genres= {'alltime': self.get_user_genres(self.listening_history),
                                        'morning': self.get_time_of_day_genres('morning'),
                                        'evening': self.get_time_of_day_genres('evening'),
                                        'recent': self.get_recent_genre_interest(),},
                    top_artists= self.get_top_artists(),
                    favorite_decade= self.get_favorite_decade(),
                    )
        return user_traits
    
    def get_user_genres(self, 
                        df_user_streams: pd.DataFrame = None, 
                        top_n: int = 5
                        ) -> list[str]:
        """
        Identifies the user's favorite genres based on their listening history. This method can analyze the entire
        history or a specific subset to determine preferred genres during particular times of day.

        Args:
            df_user_streams (pd.DataFrame, optional): A DataFrame subset of the user's listening history. If not provided,
                                                    the method will use the user's entire listening history.
            top_n (int, optional): The number of top genres to return. Defaults to 5.

        Returns:
            list: A list of the user's top_n favorite genres, sorted by frequency of appearance in their listening history.
        """
        user_genres = []

        # Add all genres listened to by the user to the list
        for song_id in df_user_streams['song_id']:
            genres = self.data.get_genres_of_song(song_id)
            # Since genres are of the form 'electronic|disco|alternative|alternative_rock'
            user_genres.extend(self.data.genres_tags_to_list(genres)) 

        # Get the most common genres
        genre_counts = Counter(user_genres)
        most_common_genres = genre_counts.most_common(top_n)

        # Get the first entry of tuple (the genre string)
        most_common_genres = [genre[0] for genre in most_common_genres]

        # if None is in the list (because some songs don't have genres), remove it
        if None in most_common_genres:
            most_common_genres.remove(None)

        # If the list is empty, add None
        if len(most_common_genres) == 0:
            most_common_genres = [None]
        
        return most_common_genres
    
    def get_time_of_day_genres(self, time_of_day: str) -> list[str]:
        """
        Determines the user's preferred genres during a specific time of day (e.g., morning or evening) by analyzing
        their listening history.

        Args:
            time_of_day (str): The time of day for which to determine genre preferences. Expected values are 'morning' or 'evening'.

        Returns:
            list: A list of the top 3 genres preferred by the user during the specified time of day.
        """
        df_listening_history = self.listening_history.copy()
        df_listening_history['hour'] = pd.to_datetime(df_listening_history['timestamp']).dt.hour
        if time_of_day == 'morning':
            time_frame = df_listening_history['hour'].between(5, 12)
        elif time_of_day == 'evening':
            time_frame = df_listening_history['hour'].between(17, 23)
        
        genres = self.get_user_genres(df_listening_history[time_frame], top_n=3)
        return genres

    def get_top_artists(self, top_n: int = 3) -> list[str]:
        """
        Retrieves the top N artists from the user's listening history, based on frequency of listens.

        Args:
            top_n (int, optional): The number of top artists to retrieve. Defaults to 3.

        Returns:
            list[str]: A list of the user's top_n artists.
        """
        artists_of_streams = self.data.get_artists_of_streams(self.listening_history) # a Series from .apply()
        top_artists = artists_of_streams.value_counts().head(top_n).index.tolist()
        return top_artists

    def get_recent_genre_interest(self, top_n: int = 3, recent_n: int = 50) -> list[str]:
        """
        Identifies the genres the user has shown the most interest in recently by analyzing their latest listening activity.

        Args:
            top_n (int, optional): The number of recent favorite genres to identify. Defaults to 3.
            recent_n (int, optional): The number of recent streams to consider for this analysis. Defaults to 50.

        Returns:
            list: A list of the user's top_n recent favorite genres.
        """
        df_recent_streams = self.listening_history.nlargest(recent_n, 'timestamp')
        recent_genres = self.get_user_genres(df_recent_streams, top_n=top_n)
        return recent_genres

    def get_favorite_decade(self):
        """
        Determines the user's favorite decade of music based on the release years of songs in their listening history.

        Returns:
            int: The most frequently listened to decade of music by the user. Defaults to 2000 if no clear preference is identified.
        """
        df_listening_history = self.listening_history.copy()
        # Get release_date year from data
        df_listening_history['release_year'] = df_listening_history['song_id'].apply(self.data.get_release_year_of_song)

        # Get decade - 2008 translates to 2000, 1999 to 1990 etc. 
        df_listening_history['release_decade'] = df_listening_history['release_year'].apply(
            lambda x: (int(x)//10)*10 if not pd.isnull(x) else None
        )

        # Get favorite decade
        favorite_decade = df_listening_history['release_decade'].value_counts().idxmax()
        if favorite_decade is None:
            favorite_decade = 2000

        return int(favorite_decade)


class UserTraits:
    """
    TLDR: Stores metadata about UserTraits in a consistent way. 
    
    Encapsulates the detailed traits and preferences of a user, including demographic information,
    favorite genres for various contexts, top artists, and preferred music decade.

    Attributes:
        user_id (int): Unique identifier of the user.
        gender (str): Gender of the user.
        yob (int): Year of birth of the user.
        favorite_genres (dict[list]): User's favorite genres categorized by different contexts ('alltime', 'morning', 'evening', 'recent').
        top_artists (list[str]): List of the user's top artists.
        favorite_decade (str): The user's favorite decade of music.
        age (int or None): Calculated age of the user, updated based on streaming activity.

    Args:
            user_id (int): The unique identifier of the user.
            gender (str): The gender of the user.
            yob (int): The year of birth of the user.
            favorite_genres (dict[list]): The user's favorite genres, organized by time context.
            top_artists (list[str]): The user's top artists.
            favorite_decade (str): The user's favorite music decade.
    """
    def __init__(self, 
                 user_id: int, 
                 gender: str, 
                 yob: int, 
                 favorite_genres: dict[list],
                 top_artists: list[str],
                 favorite_decade: str,
                ):
        self.user_id = user_id
        self.gender = gender
        self.yob = yob # year of birth
        self.favorite_genres = favorite_genres # keys are 'alltime', 'morning', 'evening', 'recent'
        self.top_artists = top_artists
        self.favorite_decade = favorite_decade
        self.age = None

    def to_dict(self) -> dict:
        """
        Converts the UserTraits instance into a dictionary for easier access and manipulation.

        Returns:
            dict: A dictionary representation of the UserTraits instance.
        """
        return self.__dict__

    def update_age(self, stream_year: int):
        """
        Updates the user's age based on the year of a specific stream, allowing for dynamic age calculation
        as new streaming data is analyzed.

        Args:
            stream_year (int): The year of the stream to use for age calculation.
        """
        if type(stream_year) is not int:
            self.age = None
        else:
            self.age = stream_year - self.yob
