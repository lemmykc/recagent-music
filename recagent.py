from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from textwrap import dedent
from langchain.schema import HumanMessage, SystemMessage
from typing import TYPE_CHECKING
from song import Song
from dotenv import load_dotenv
import os

if TYPE_CHECKING:
    from user import UserTraits
#from data import Song, UserTraits

class RecAgent():
    """
    TLDR: All querying and processing of the OpenAI API is done in this class.

    Represents a recommender agent (in the end, not really an agent, but the terminology persists) that utilizes ChatGPT to predict 
    whether a user will play or skip a recommended song. The agent takes into account the user's traits, their recent song 
    listening history, and information about the recommended song. 

    The agent is designed to be stateless, with no persistent storage of data or state between predictions, ensuring that
    each prediction is independent of previous ones. This design choice simplifies the use of the agent in various scenarios
    without the need for managing state across sessions. 

    (It only takes about 0.012 seconds to instantiate this class, so it's not a big deal to do so.)

    NOTE: The openai_api_key is stored in a .env file, and is accessed using the os module.

    Attributes:
        llm (ChatOpenAI): An instance of the ChatOpenAI class from langchain, configured for interacting with the OpenAI API.
    """
    def __init__(self):
        self.llm=ChatOpenAI(
            max_tokens=1500,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("MODEL_NAME", "gpt-3.5-turbo-16k"),
            max_retries=100
        )


    ### GENERATING FUNCTIONS ###
        
    def predict_play_or_skip(self, 
                             user_traits: 'UserTraits', 
                             song_history: list[Song], 
                             recommended_song: Song, 
                             lyric_option: str ="chatgpt_memory_summary",
                             user_profile_option: str ="simple"
                             ) -> tuple[str, str, str]:
        """
        Predicts whether a user will play or skip a recommended song based on their traits, their recent listening history,
        and details of the recommended song.

        Args:
            user_traits (UserTraits): User traits including age, gender, and preferred genres.
            song_history (list): A list of Song objects representing the user's recent listening history.
            recommended_song (Song): Information about the recommended song.
            lyric_option (str): A hyperparameter that specifies how lyrics are summarized for 
                                inclusion in the prompt. Defaults to 'chatgpt_memory_summary'.
            user_profile_option (str): A hyperparameter that specifies the complexity of the user profile 
                                        included in the prompt. Defaults to 'simple'.

        Returns:
            tuple: A tuple containing the generated prompt, the predicted action ('play' or 'skip'), and ChatGPT's response.
        """
        # Generate the current state prompt
        prompt = self.generate_prediction_prompt(user_traits, song_history, recommended_song, lyric_option, user_profile_option)

        # Get response from ChatGPT
        messages = [HumanMessage(content=prompt)]
        response = self.llm(messages).content

        # Get the action from the response
        action = self.get_action_from_response(response)

        return prompt, action, response
    
    def generate_prediction_prompt(self, 
                                user_traits: 'UserTraits', 
                                song_history: list[Song], 
                                recommended_song: Song, 
                                lyric_option: str, 
                                user_profile_option: str
                                ) -> str:
        """
        Generates a text prompt for ChatGPT based on the user's traits, their listening history, and the recommended song.

        Args:
            user_traits (UserTraits): Information about the user, including traits like age and preferred genres.
            song_history (list): The user's recent song listening history, Song objects.
            recommended_song (Song): Details of the song being recommended.
            lyric_option (str): Determines how song lyrics are summarized in the prompt.
            user_profile_option (str): Specifies the detail level for the user profile in the prompt.

        Returns:
            str: The generated prompt ready to be sent to ChatGPT.
        """
        # Get formatted information for the prompt
        formatted_user_traits = self.format_user_traits(user_traits, simple_or_expanded = user_profile_option)
        formatted_song_list = self.generate_formatted_song_list(song_history, lyric_option)
        recommended_song_info = self.format_song_info(recommended_song, include_action=False, lyric_option=lyric_option)
        period_of_day = recommended_song.period_of_day_listening
       
        # Create the prompt template        
        recommendation_prompt_template = PromptTemplate(template=dedent("""
            {user_traits_description} They have just listened to the following songs in the {period_of_day}:\n
            {formatted_song_list}

            The Deezer recommender has just recommended the next song:
            {recommended_song_info}

            Does the user decide to play or skip the next recommended song? 
            If play, please respond only with [PLAY], or if they skip please respond only with [SKIP].
        """),
        input_variables=["user_traits_description", "period_of_day", "formatted_song_list", "recommended_song_info"],
        )

        # Fill in the prompt template
        prompt = recommendation_prompt_template.format(
            user_traits_description=formatted_user_traits,
            period_of_day=period_of_day,
            formatted_song_list=formatted_song_list,
            recommended_song_info=recommended_song_info
        )

        return prompt
    
    def get_chatgpt_memory_summary(self, song: Song) -> str:
        """
        Generates a summary of a song based solely on ChatGPT's internal knowledge, without using actual lyrics.

        Args:
            song (Song): The song object to summarize.

        Returns:
            str: A summary of the song, or None if ChatGPT cannot provide a meaningful summary.
        """

        # Initialise the system and user messages
        system_content = """You are a knowledgeable assistant with expertise in music analysis. 
                            Provide detailed, single-sentence analyses of songs that encompass not only the lyrical themes 
                            but also the emotional mood, genre, historical context, and musical composition. 
                            Highlight how these elements interplay to create the song's unique character. 
                            If a song is not recognized or if you can't provide an analysis, respond with 'UNKNOWN'.
                            If it is purely instrumental and has no lyrics, please ensure you do not speak about lyrics and mention that it is instrumental in your response.
                            """
        user_content = f"Analyse '{song.name}' by {song.artist}, focusing on its lyrical themes, musical style, and overall mood, in only one sentence."
        messages = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=user_content),
                ]

        # Run the prompt through ChatGPT to get summary
        summary = self.llm(messages).content

        # Turn into None if UNKNOWN 
        summary = self.handle_bad_summary(summary)

        return summary
    
    def get_chatgpt_scraped_lyrics_summary(self, song: Song) -> str:
        """
        Requests ChatGPT to summarize a song based on its scraped lyrics.

        Args:
            song (Song): The song object, including scraped lyrics, to summarize.

        Returns:
            str: A summary of the song based on its lyrics, or None if the summary is not available or meaningful.
        """
        # Initialise the system and user messages
        system_content = """You are a knowledgeable assistant with expertise in music analysis. 
                            Provide detailed, single-sentence analyses of song lyrics that encompass thematic material
                            as well as emotional mood and genre.
                            Highlight how these elements interplay to create the song's unique character. 
                            If it appears the lyrics are broken in some way, respond with 'UNKNOWN'.
                            """
        
        user_content = f"Analyse the following lyrics to the song '{song.name}' by {song.artist} providing a single sentence summary:\n\n{song.song_summaries['lyrics']}"
        messages = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=user_content),
                ]

        # Get the summary from ChatGPT
        summary = self.llm(messages).content

        # Turn into None if UNKNOWN
        summary = self.handle_bad_summary(summary)

        return summary

    def format_song_info(self, song: Song, 
                         include_action: bool = True, 
                         lyric_option: str = "chatgpt_memory_summary") -> str:
        """
        Formats detailed information about a song into a consistent, readable string, optionally including the user's action
        (depending on whether it's a recommended song or not).

        Args:
            song (Song): The song object containing metadata to be formatted.
            include_action (bool, optional): Whether to include the user's action (play or skip) in the formatted string. Defaults to True.
            lyric_option (str, optional): Determines how song lyrics are summarized. 

        Returns:
            str: A formatted string containing the song's details, such as name, artist, genres, release year, duration,
                and optionally, the lyric summary and user's action.
        """

        # Format the genres list into a readable string
        genres_text = self.format_genres_list_to_text(song.genres)
        
        # Build the basic song info
        song_info = (f"{song.name} by {song.artist}, classified under genres {genres_text}, "
                    f"which was released in the year {song.year}, "
                    f"which goes for a duration of {song.length} seconds.")

        # Include the lyric summary if requested
        if lyric_option == "no_lyric_summary":
            pass
        else:
            summary = song.song_summaries[lyric_option]
            if lyric_option=="first_n_lyric_lines":
                song_info += f" The first few lines of the song are '{summary}'."
            else:
                song_info += f" The summary is '{summary}'."

        # Include the user action if requested
        if include_action:
            # Append the action information without introducing extra spaces
            song_info += f" The user {self.past_tense_action(song.action)}.\n"

        return song_info
        

    ### HELPER FUNCTIONS ###
    
    def format_genres_list_to_text(self, genres_list: list) -> str:
        """
        Formats a list of genres into a readable string ("rock, pop and classical"), handling various cases such as 
        empty lists, single genres, or multiple genres.

        Args:
            genres_list (list): A list of genre strings to be formatted.

        Returns:
            str: A formatted string representing the genres, correctly comma-separated and with 'and' before 
                the last genre if multiple.
        """

        # Clean the list by removing any duplicate entries
        genres_list = list(set(genres_list))

        # Replace None with "None"
        for i in range(len(genres_list)):
            if genres_list[i] is None:
                genres_list[i] = "None"

        # Check if the list is empty
        if not genres_list:
            formatted_genres = ""

        # If there's only one genre, return it
        elif len(genres_list) == 1:
            formatted_genres = genres_list[0]

        # If there are two genres, join them with 'and'
        elif len(genres_list) == 2:
            formatted_genres = ' and '.join(genres_list)

        # If there are more than two genres, join all but the last with commas, and add the last with 'and'
        else:
            formatted_genres = ', '.join(genres_list[:-1]) + ' and ' + genres_list[-1]

        # Replace underscores with spaces
        formatted_genres = formatted_genres.replace('_', ' ')

        return formatted_genres
    
    def format_user_traits(self, user_traits: 'UserTraits', simple_or_expanded: str = 'simple') -> str:
        """
        Formats a user's traits into a detailed description, suitable for inclusion in prompts to the language model.

        Args:
            user_traits (UserTraits): An object containing the user's traits, such as age, gender, and favorite genres.
            simple_or_expanded (str, optional): Specifies whether to format the traits in a "simple" or "expanded" 
                                                detail level. Defaults to 'simple'.

        Returns:
            str: A string describing the user's traits, formatted according to the specified detail level.
        """
        # Format favorite genres for different times and recent interests
        formatted_genres_alltime = self.format_genres_list_to_text(user_traits.favorite_genres['alltime'])
        formatted_genres_morning = self.format_genres_list_to_text(user_traits.favorite_genres['morning'])
        formatted_genres_evening = self.format_genres_list_to_text(user_traits.favorite_genres['evening'])
        formatted_genres_recent = self.format_genres_list_to_text(user_traits.favorite_genres['recent'])
        
        # Format top artists
        formatted_top_artists = self.format_genres_list_to_text(user_traits.top_artists)

        # Format the user traits based on the detail level
        if simple_or_expanded == 'simple':
            user_traits_info = f"The user is a {user_traits.age} year old {user_traits.gender} who likes {formatted_genres_alltime} music."
        else:
            user_traits_info = (
                f"The user is a {user_traits.age} year old {user_traits.gender}, "
                f"who typically enjoys {formatted_genres_alltime} music. "
                f"In the mornings, they prefer {formatted_genres_morning}, "
                f"while in the evenings, they lean towards {formatted_genres_evening}. "
                f"Recently, they've shown more interest in {formatted_genres_recent}. "
                f"Their top artists include {formatted_top_artists}, "
                f"and they have a particular fondness for music from the {user_traits.favorite_decade}s."
            )

        return user_traits_info

    def past_tense_action(self, action: str) -> str:
        """
        Converts a user's action into past tense for inclusion in formatted song information.

        Args:
            action (str): The user's action with the song, expected to be either 'play' or 'skip'.

        Returns:
            str: The past tense form of the user's action, suitable for narrative descriptions.
        """
        if action=='play':
            return 'played'
        elif action=='skip':
            return 'skipped'
        else:
            return "did something unknown with."
        
    def generate_formatted_song_list(self, song_history: list[Song], lyric_option: str) -> str:
        """
        Generates a formatted list of songs from the user's listening history, including details and optionally lyrics summaries.

        Args:
            song_history (list[Song]): A list of Song objects representing the user's recent listening history.
            lyric_option (str): Specifies how to include song lyrics in the formatted list.

        Returns:
            str: A newline-separated string list of formatted song details from the user's listening history.
        """
        song_list = "\n".join([
            self.format_song_info(song, lyric_option=lyric_option) for song in song_history
        ])
        return song_list
    
    def get_action_from_response(self, response: str) -> str:
        """
        Extracts the predicted action (play or skip) from the language model's response.

        Args:
            response (str): The raw response string from the language model.

        Returns:
            str: The extracted action ('play' or 'skip'), or None if the response is ambiguous or does not clearly contain either action.
        """
        if 'PLAY' in response and 'SKIP' not in response:
            action = 'play'
        elif 'SKIP' in response and 'PLAY' not in response:
            action = 'skip'
        else:
            # This could simply be an unidentifiable response, 
            # or it could be that the user has responded with both 'PLAY' and 'SKIP' in their response.
            action = None
        return action
    
    @staticmethod
    def handle_bad_summary(summary: str) -> str:
        """
        Checks if a song summary is marked as "UNKNOWN" and converts it to None for consistent handling of unavailable summaries, 
        useful for data processing in the dataframe.

        Args:
            summary (str): The song summary returned by the language model.

        Returns:
            str or None: None if the summary is "UNKNOWN", otherwise returns the original summary.
        """
        if "UNKNOWN" in summary:
            summary = None
        return summary

