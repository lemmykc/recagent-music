from bs4 import BeautifulSoup, Comment
from urllib.request import urlopen
from typing import TYPE_CHECKING
#from recagent import RecAgent
from typing import TYPE_CHECKING
from unidecode import unidecode
from pandas import Timestamp
if TYPE_CHECKING:
    from recagent import RecAgent

class Song:
    """
    TLDR: Stores Song metadata and handles scraping of azlyrics.com for lyrics. 

    Represents a song with attributes such as song ID, name, artist, genres, length, year of release,
    and the user's action (play or skip). This class also handles the retrieval of song lyrics and the generation of
    various summaries based on the song's lyrics and its metadata. The Song class is designed to encapsulate all 
    relevant information and functionalities associated with a song.

    The Song class is capable of generating summaries using a recommender agent (`RecAgent`), which can interact with
    language models like ChatGPT to produce summaries based on the song's intrinsic attributes or its lyrics. Summaries
    can include a general memory-based summary, a summary based on scraped lyrics, or the first few lines of the lyrics
    themselves. The class also determines the period of the day when the song was listened to, based on the provided
    timestamp.

    Attributes:
        song_id (int): Unique identifier for the song.
        name (str): Name of the song.
        artist (str): Artist or group that performed the song.
        genres (list): List of genres associated with the song.
        length (int): Duration of the song in seconds.
        year (int): Release year of the song.
        action (str): User's action regarding this song ('play' or 'skip').
        listen_timestamp (Timestamp): Timestamp indicating when the song was listened to.
        n_lyric_lines (int): Number of lyric lines to include in summaries (defaults to 4).
        period_of_day_listening (str): Period of the day ('morning', 'afternoon', 'evening') when the song was listened to, derived from `listen_timestamp`.
        song_summaries (dict): Container for storing various types of song summaries and lyrics.
    """
    def __init__(self, 
                 song_id: int, 
                 name: str, 
                 artist: str, 
                 genres: list, 
                 length: int, 
                 year: int, 
                 action: str,
                 listen_timestamp: Timestamp,
                 n_lyric_lines: int = 4):
        self.song_id = song_id
        self.name = name
        self.artist = artist
        self.genres = genres
        self.length = length
        self.year = year
        self.action = action
        self.listen_timestamp = listen_timestamp
        self.period_of_day_listening = self.get_period_of_day_listening()
        self.song_summaries = {'lyrics': None,
                                'first_n_lyric_lines': None,
                                'chatgpt_memory_summary': None,
                                'chatgpt_scraped_lyrics_summary': None}
        self.n_lyric_lines = n_lyric_lines

    def generate_summaries(self, 
                           recagent: 'RecAgent', 
                           lyric_options: list = ['no_lyric_summary']
                           ):
        """
        Generates and stores summaries and lyrics for the song based on specified options.

        Args:
            recagent (RecAgent): The recommender agent instance used to generate summaries.
            lyric_options (list, optional): A list specifying which types of summaries to generate. Options include
                                            'first_n_lyric_lines', 'chatgpt_memory_summary', and 'chatgpt_scraped_lyrics_summary'.
                                            Defaults to ['no_lyric_summary'], which skips summary generation.
        """
        
        # If no lyric summary is requested, skip summary generation, otherwise, generate the requested summaries
        if "first_n_lyric_lines" in lyric_options or "chatgpt_scraped_lyrics_summary" in lyric_options:
            self.song_summaries['lyrics'] = self.get_lyrics()
            self.song_summaries['first_n_lyric_lines'] = self.get_first_n_lines(self.song_summaries['lyrics'], self.n_lyric_lines)

        if "chatgpt_memory_summary" in lyric_options:
            self.song_summaries['chatgpt_memory_summary'] = recagent.get_chatgpt_memory_summary(self)

        if "chatgpt_scraped_lyrics_summary" in lyric_options:
            self.song_summaries['chatgpt_scraped_lyrics_summary'] = recagent.get_chatgpt_scraped_lyrics_summary(self)
    
    def to_dict(self) -> dict:
        """
        Converts the Song instance into a dictionary, preserving all instance attributes.

        Returns:
            dict: A dictionary representation of the Song instance, including all its attributes.
        """
        return self.__dict__

    def get_period_of_day_listening(self) -> str:
        """
        Determines the period of the day during which the song was listened to based on the listen timestamp.

        Returns:
            str: A string indicating the period of the day ('morning', 'afternoon', 'evening') when the song was listened to.
        """
        # In case something goes wrong, default to 'evening'
        try: 
            hour = self.listen_timestamp.hour
        except: 
            hour = 18

        # Determine the period of the day
        if hour < 12:
            return 'morning'
        elif hour < 17:
            return 'afternoon'
        else:
            return 'evening'
    
    def get_lyrics(self):
        """
        Attempts to retrieve the lyrics of the song from an external source using web scraping techniques.

        Returns:
            str or None: The lyrics of the song if successfully retrieved, otherwise None.
        """
        try:
            # Generate the URL for the song on azlyrics.com, retrieve the HTML, parse where the lyrics should be
            song_url = self.generate_azlyrics_url(self.name, self.artist)
            song_html = urlopen(song_url)
            soup = BeautifulSoup(song_html, "html.parser")
            comment = soup.find(string=lambda x: isinstance(x, Comment) and "Usage of azlyrics.com content" in x)

            # Extract the lyrics from the HTML. (ChatGPT helped me write this, it's a bit hard to parse but does the job).
            lyrics = ""
            if comment:
                element = comment.next_element
                while element and element.name != "div":
                    if not isinstance(element, Comment):
                        lyrics += str(element)
                    element = element.next_element

            # Clean the lyrics and return them.
            clean_lyrics = lyrics.replace('<br/>', '').strip()
            return clean_lyrics if clean_lyrics else None
        
        except Exception:
            return None
    
    @staticmethod
    def generate_azlyrics_url(song: str, artist: str) -> str:
        """
        Constructs a URL for querying song lyrics from AZLyrics based on the song and artist names.

        NOTE - This function is not perfect and may not work for all songs. It's a simple heuristic based on trial and error.

        Args:
            song (str): The name of the song.
            artist (str): The name of the artist.

        Returns:
            str: A URL string pointing to the AZLyrics page for the song's lyrics.
        """
        # Remove "The " from the beginning or ", The" from the end of the artist's name
        artist = artist.lower()
        if artist.startswith("the "):
            artist = artist[4:]
        elif artist.endswith(", the"):
            artist = artist[:-5]

        # If "(feat. )" is in the song name, remove it and everything after it
        song = song.lower()
        contains_remix = "(remix)" in song
        if "(feat. " in song:
            song = song[:song.index("(feat. ")]
        if "(ft. " in song:
            song = song[:song.index("(ft. ")]

        # If "(remast" is in the song name, remove it and everything after it
        if "(remast" in song:
            song = song[:song.index("(remast")]
            
        # Add the remix back in for e.g. 'Or Nah (feat. The Weeknd, Wiz Khalifa & DJ Mustard) (Remix)'
        if contains_remix:
            song += " remix"
        
        # If "(live" is in the song name, remove it and everything after it
        if "(live" in song:
            song = song[:song.index("(live")]

        # If "(feat. )" is in the artist name, remove it and everything after it
        if "(feat. )" in artist:
            artist = artist[:artist.index("(feat. ")]
        if "(ft. " in artist:
            artist = artist[:artist.index("(ft. ")]

        # Replace any accented character with its unaccented equivalent (since a lot of French songs)
        song = unidecode(song)
        artist = unidecode(artist)

        # Replace f**k with fuck
        for f_word_variant in ["f*ck", "f**k", "f***"]:
            if f_word_variant in song:
                song = song.replace(f_word_variant, "fuck")
            
        # Replace $ character with s
        if "$" in song:
            song = song.replace("$", "s")
        
        # Remove punctuation, convert to lowercase, and replace spaces with hyphens
        artist_clean = ''.join(c for c in artist if c.isalnum())
        song_clean = ''.join(c for c in song if c.isalnum()).lower()
        url = f"https://www.azlyrics.com/lyrics/{artist_clean}/{song_clean}.html"

        return url
    
    @staticmethod
    def get_first_n_lines(text: str, n: int) -> str:
        """
        Extracts the first n lines from a given text string. Mainly used for getting the first n lines of lyrics.

        Args:
            text (str): The text from which to extract lines.
            n (int): The number of lines to extract.

        Returns:
            str or None: A string containing the first n lines of the given text, or None if the text is None.
        """
        if text is None:
            return None
        return ' '.join(text.split('\n')[:n])

