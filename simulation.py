from typing import Any, Dict, List, Optional
from user import User
from data import Data
from recagent import RecAgent
import pandas as pd
from utils import directory_creator
import json
from tqdm import tqdm
from scipy.stats import binomtest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from user import UserTraits

class Simulation:
    """
    TLDR: This is the main class for running simulations of ChatGPT's music recommendation performance.

    A simulation class for evaluating ChatGPT's music recommendation performance. The class sets up and runs simulations
    based on various configurations, including the length of listening history, the number of recommendations per trial,
    and the use of different lyric summary options.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the simulation, including simulation name, experiment name, 
                                 number of trials, length of listening history, number of recommendations, and data settings.
        manager (SimulationManager): Manages logging and results of the simulation.
        data (Data): Data handler for accessing user, song, and streaming information.
        recagent (RecAgent): The class that is responsible for all prompts to ChatGPT (the llm is instantiated here).
        debug (bool): Print the prompt and response on each trial for debugging purposes.
        lyric_options (List[str]): Specifies the lyric summary options to use in the simulation.
        seen_songs_cache (Dict[int, Song]): A cache to store song instances (and therefore their summaries) to avoid recomputation.

    Args:
        name (str): Name of the simulation for identification.
        num_trials (int): Number of trials to run in the simulation.
        length_history_sequence (int): Length of user's listening history considered in each trial.
        num_recommendations (int, optional): Number of recommendations to test per trial. Defaults to 1.
        only_smartradio (bool, optional): If True, only uses the smartradio dataset for simulation. Defaults to False.
        lyric_options (List[str], optional): Specifies which lyric options to use. Defaults to ["chatgpt_memory_summary"].
        user_profile_options (List[str], optional): Specifies which user profile options to consider. Defaults to ['simple'].
        n_lyric_lines (int, optional): Number of lyric lines to consider. Defaults to 4.
        exp_name (str, optional): Name of the experiment version. Defaults to "EXP001".
        debug (bool, optional): If True, enables debug logging. Defaults to False.
        big_or_small_data (str, optional): Chooses between 'big' or 'small' dataset configurations. Defaults to "small".
    """

    def __init__(self, 
                 name: str,
                 num_trials: int, 
                 length_history_sequence: int,
                 num_recommendations: int = 1,
                 only_smartradio: bool = False,
                 lyric_options: list[str] = ["chatgpt_memory_summary"],
                 user_profile_options: list[str] = ['simple'],
                 n_lyric_lines: int = 4,
                 exp_name: str = "EXP001",
                 debug: bool = False,
                 big_or_small_data: str = "small" 
                 ): 
        
        # Setup config and simulation manager 
        self.config = {'name': name,
                    'exp_name': exp_name,
                    'num_trials': num_trials,
                    'length_history_sequence': length_history_sequence, # The length of listening history to be used as context.
                    'num_recommendations': num_recommendations,
                    'n_lyric_lines': n_lyric_lines,
                    'only_smartradio': only_smartradio,
                    'lyric_options': lyric_options,
                    'user_profile_options': user_profile_options,
                    'big_or_small_data': big_or_small_data
                    }
        self.manager = SimulationManager(config=self.config)
        
        # Load the data
        if big_or_small_data == "small":
            catalog_path = 'simcares_data/simcares_catalog.csv'
            streams_path = 'simcares_data/simcares_streams.csv'
            users_path = 'simcares_data/simcares_users.csv'
        elif big_or_small_data == "big":
            catalog_path = 'simcares_data/simcares_20FEB2024_catalog.csv'
            streams_path = 'simcares_data/simcares_20FEB2024_streams.csv'
            users_path = 'simcares_data/simcares_20FEB2024_users.csv'
        self.data = Data(catalog_path = catalog_path,
                         streams_path=streams_path,
                         users_path=users_path,
                         min_stream_history=length_history_sequence + num_recommendations,
                         only_smartradio=only_smartradio,
                         )
        
        # Instantiate the recagent
        self.recagent = RecAgent()
        
        # Misc other things
        self.debug = debug
        self.lyric_options = lyric_options
        self.seen_songs_cache = {} # keys are song_id

        # Automatic stopping criteria
        self.scraping_failure_threshold = 10
        self.scraping_failure_counter = 0
        

    ### SIMULATION FUNCTIONS ###

    def run_simulation(self):
        """
        Main method to execute the simulation over the configured number of trials. It iteratively simulates user interactions
        with recommended songs, evaluates ChatGPT's recommendations based on different lyric summary and user profile options, and logs the results.

        This method handles the setup for each trial, including selecting a random user, fetching their listening history,
        generating recommendations, and evaluating the success of these recommendations. It also implements automatic stopping criteria
        based on consecutive scraping failures, to prevent excessive resource usage in case of external API limitations or errors.
        
        No parameters are required as it uses the instance's configuration. Results are logged via the SimulationManager instance.
        """

        # Iterate through trials with progress bar in tqdm
        for i in tqdm(range(self.config['num_trials']), desc='Running Simulation'):
            # Before each trial, check if there have been 10 consecutive failures
            if self.scraping_failure_counter >= self.scraping_failure_threshold:
                print(f"Stopping simulation due to {self.scraping_failure_threshold} consecutive trials with 100% lyric scraping failure (AZLyrics is not happy).")
                break

            # Set up random user
            rand_user_id = self.data.get_random_user()
            user = User(rand_user_id, self.data)
            user_traits = user.user_traits

            # Get a random subset of their listening history and next recommended song(s)
            song_history, recommended_songs = user.get_rand_subset_songs_history(self.config['length_history_sequence'], 
                                                                                self.config['num_recommendations'],
                                                                                n_lyric_lines = self.config['n_lyric_lines'])
            
            # Generat the lyrics if the song hasn't been seen before, otherwise get from cache 
            song_history = self.update_song_summaries(song_history)
            recommended_songs = self.update_song_summaries(recommended_songs)

            # Ask ChatGPT for its predicted action 
            for lyric_option in self.config['lyric_options']: 
                # Iterating over lyric options goes here so that we can compare performance on the exact same sample 
                # to see whether increased context makes predictions better or neutral. 
                for user_profile_option in self.config['user_profile_options']:
                    # Ditto above comment.
                    self.run_trial(user_traits, 
                                    song_history, 
                                    recommended_songs, 
                                    lyric_option,
                                    user_profile_option,
                                    trial_id = i)
                    
            # Check for 100% lyric scraping failure in this trial
            trial_failure = self.check_lyric_scraping_failure()
            if trial_failure:
                self.scraping_failure_counter += 1
            else:
                self.scraping_failure_counter = 0
                
        # Finalise results
        self.manager.finalise_results()

        # Print results summary to terminal (this was turned into a staticmethod so that we can use it on a loaded CSV)
        self.manager.print_hyperparameter_comparison_results(self.manager.results_df)

    def run_trial(self, user_traits: 'UserTraits', 
                  original_song_history: list, 
                  recommended_songs: list, 
                  lyric_option: str,
                  user_profile_option: str,
                  trial_id: int):
        """
        Executes a single trial of the simulation, simulating the recommendation process for a user based on their traits,
        listening history, and the songs recommended. 

        Each trial involves predicting the user's action for each recommended song, logging the results, and updating the user's song history.

        Args:
            user_traits (UserTraits): An instance of UserTraits containing attributes of the user, such as ID, gender, age, favorite genres, etc.
            original_song_history (list): A list of Song instances representing the user's recent listening history of length length_history_sequence.
            recommended_songs (list): A list of Song instances representing the songs recommended to the user of length num_recommendations.
            lyric_option (str): The lyric summary option to be used for this trial.
            user_profile_option (str): The user profile option to be used for this trials
            trial_id (int): The identifier for the current trial, used for logging purposes.

        This method logs the trial's results, including predictions and actual user actions, to the SimulationManager. It also
        handles the dynamic adjustment of the user's song history based on the simulation's progression.
        """

        # Create a copy of the original song history for this trial to prevent modifying the original list
        song_history = original_song_history.copy()

        # Calculate missing summaries percentages
        perc_missing_chatgpt_memory, perc_missing_first_n_lines = self.calculate_missing_summaries_percents(original_song_history, recommended_songs)
                
        # Ask ChatGPT for its predicted action for each recommended song, iteratively adding each recommended song to the next history sequence
        for song_index in range(len(recommended_songs)):
            recommended_song = recommended_songs[song_index]

            # Prompt ChatGPT to predict whether the given user plays or skips the next recommended song. 
            prompt, predicted_action, response = self.recagent.predict_play_or_skip(user_traits, 
                                                                                    song_history, 
                                                                                    recommended_song, 
                                                                                    lyric_option,
                                                                                    user_profile_option)
            
            # Log prompt and response text
            prediction_text = f'ChatGPT predicted {predicted_action}. The correct answer was {recommended_song.action}.'
            output_text = f"trial_id={trial_id}, idx_rec={song_index}, lyric_option={lyric_option}, user_profile_option={user_profile_option}\n\nPROMPT\n{prompt}\nRESPONSE\n{response}\n\n{prediction_text}\n\n" + "-"*80
            self.manager.log_text(output_text)

            # Print the prompt and response if debug is enabled
            if self.debug:
                print(output_text)

            # Log the trial results
            trial_results = {
                'trial_id': trial_id,
                'user_id': user_traits.user_id,
                'user_gender': user_traits.gender,
                'user_age': user_traits.age,
                'user_genres_alltime': user_traits.favorite_genres['alltime'],
                'user_genres_morning': user_traits.favorite_genres['morning'],
                'user_genres_evening': user_traits.favorite_genres['evening'],
                'user_genres_recent': user_traits.favorite_genres['recent'],
                'user_top_artists': user_traits.top_artists,
                'user_favorite_decade': user_traits.favorite_decade,
                'song_history': [song.to_dict() for song in song_history],  # Assuming a to_dict method exists
                'recommended_song': recommended_song.to_dict(),
                'idx_recommended_song': song_index, # i.e. the "token position", sort of, not the song id itself. 
                'lyric_option': lyric_option,
                'user_profile_option': user_profile_option,
                'actual_action': recommended_song.action,
                'predicted_action': predicted_action,
                'perc_missing_chatgpt_memory': perc_missing_chatgpt_memory,
                'perc_missing_lyrics': perc_missing_first_n_lines,
            }
            self.manager.log_trial_to_df(trial_results)

            # Append recommended song to song history list for next iteration
            song_history.append(recommended_song)

    
    ### HELPER FUNCTIONS ###

    def update_song_summaries(self, song_list: list):
        """
        Updates or retrieves the summaries for a list of songs. It checks if summaries are already generated and cached;
        if not, it generates new summaries. This is used to minimize redundant processing and ChatGPT API calls.

        NOTE - The Song class is not agnostic of the user's interaction with the Song. The specific user's action and timestamp are
        contained in Song. Therefore, it is important that this function merely _updates_ with previously seen summaries, as opposed
        to drawing from old instances of Song. 

        Args:
            song_list (list[Song]): A list of Song instances for which to update or retrieve summaries.

        Returns:
            list: The updated list of Song instances with their summaries either retrieved from cache or newly generated.
        """
        for j in range(len(song_list)):
            song = song_list[j]
            if song.song_id not in self.seen_songs_cache:
                # Generate the song summaries in its instance
                song.generate_summaries(lyric_options=self.lyric_options, recagent = self.recagent)

                # Add the song to the cache
                self.seen_songs_cache[song.song_id] = song

                # Update corresponding entry in song_list, now with lyric summaries
                song_list[j] = song

            else:
                # Get the song summary from the cache but PRESERVE THE CURRENT ACTION (i.e. the action from the user's history in question)
                song_list[j].song_summaries = self.seen_songs_cache[song.song_id].song_summaries

        return song_list
    
    def calculate_missing_summaries_percents(self, song_history: list, recommended_songs: list):
        """
        Calculates the percentages of missing 'chatgpt_memory_summary' (because it didn't know about the song) 
        and 'first_n_lyric_lines' summaries (because the AZLyrics scrape did not work) across both
        the user's song history and the recommended songs. This is used for the automatic stopping criteria. 

        Args:
            song_history (list): A list of Song instances representing the user's listening history.
            recommended_songs (list): A list of Song instances recommended to the user.

        Returns:
            tuple: A pair of integers representing the percentage of missing 'chatgpt_memory_summary' and
                'first_n_lyric_lines' (and therefore 'lyrics') summaries, respectively.
        """
        combined_songs_list = song_history + recommended_songs
        total_songs = len(combined_songs_list)
        missing_chatgpt_memory = 0
        missing_first_n_lines = 0

        for song in combined_songs_list:
            if 'chatgpt_memory_summary' in self.lyric_options:
                if song.song_summaries['chatgpt_memory_summary'] is None:
                    missing_chatgpt_memory += 1
            if 'first_n_lyric_lines' in self.lyric_options:
                if song.song_summaries['first_n_lyric_lines'] is None:
                    missing_first_n_lines += 1

        perc_missing_chatgpt_memory = (missing_chatgpt_memory / total_songs) * 100 if total_songs > 0 else 0
        perc_missing_first_n_lines = (missing_first_n_lines / total_songs) * 100 if total_songs > 0 else 0

        perc_missing_chatgpt_memory = int(perc_missing_chatgpt_memory)
        perc_missing_first_n_lines = int(perc_missing_first_n_lines)

        return perc_missing_chatgpt_memory, perc_missing_first_n_lines
    
    def check_lyric_scraping_failure(self):
        """
        Checks if the most recent trial resulted in a 100% failure rate for lyric scraping. This is part of the simulation's
        automatic stopping criteria to avoid continued execution under conditions where necessary data cannot be obtained.

        Returns:
            bool: True if the last trial had a 100% lyric scraping failure rate, otherwise False.
        """
        if self.manager.results:
            last_trial = self.manager.results[-1]
            if last_trial['perc_missing_lyrics'] == 100:
                return True
        return False


class SimulationManager:
    """
    TLDR: Deals with config management, results logging, etc. 

    Manages the simulation lifecycle, including setup, logging, result analysis, and configuration management for the Simulation class. 
    This class is responsible for creating necessary directories for storing
    results and logs, saving simulation configurations, and providing utilities for logging both textual data and structured
    results data. It also includes functions for finalizing and summarizing results, as well as comparing the performance
    of different simulation hyperparameters.

    Attributes:
        config (Dict[str, Any]): A dictionary containing the configuration settings for the simulation. These settings
                                 include the name of the simulation, the experiment name, and paths for saving logs and results.
        results (List[Dict[str, Any]]): A list to accumulate the detailed results of each trial run within the simulation.
                                        Each entry is a dictionary representing the outcomes of a single trial, later concatenated into a dataframe.
        results_df (pd.DataFrame): A dataframe containing the structured results of the simulation, including trial outcomes and metadata.
        text_log_path (str): The file path to which detailed textual logs (e.g., prompts, responses) are written.
        results_path (str): The file path where the CSV file containing structured results of the simulation is saved.
        config_path (str): The file path where the simulation configuration is saved in JSON format.

    Args:
        config (Dict[str, Any]): A dictionary containing the configuration settings for the simulation. These settings
                                 include the name of the simulation, the experiment name, and paths for saving logs and results, 
                                 instantiated in the Simulation class.
    """
    def __init__(self, config: Dict[str, Any]):
        # Initialise attributes and create directories
        self.config = config
        self.results = []
        exp_dir = directory_creator("experiments", self.config['exp_name'])
        results_dir = directory_creator(exp_dir, "results")
        text_logs_dir = directory_creator(exp_dir, "text_logs")
        config_dir = directory_creator(exp_dir, "configs")
        self.text_log_path = f"{text_logs_dir}/{self.config['name']}_text_logs.txt"
        self.results_path = f"{results_dir}/{self.config['name']}_results.csv"
        self.config_path = f"{config_dir}/{self.config['name']}_config.json"

        # Save config to json
        self.save_config()

        # Clear existing text log file to start fresh for this simulation
        open(self.text_log_path, 'w').close()

    
    def log_trial_to_df(self, trial_data: Dict[str, Any]):
        """
        Logs the data from a single trial to the results list. This data is intended to be converted into a DataFrame
        for analysis at the end of the simulation. To avoid a huge CSV file, we remove the song summaries (since they 
        are in the text log). 

        Args:
            trial_data (Dict[str, Any]): A dictionary containing the details and outcomes of a single trial, including
                                        user information, song recommendations, and the actions taken. Each instance of trial_data 
                                        has the same keys, and the values are the results of the trial.
        """
        # Simplify song_history by excluding 'song_summaries'
        trial_data['song_history'] = [
            {key: value for key, value in song.items() if key != 'song_summaries'}
            for song in trial_data['song_history']
        ]

        # Simplify recommended_song in the same way
        trial_data['recommended_song'] = {
            key: value for key, value in trial_data['recommended_song'].items() if key != 'song_summaries'
        }

        # Apply timestamp conversion 
        trial_data['song_history'] = [self.convert_timestamp_to_string(song) for song in trial_data['song_history']]
        trial_data['recommended_song'] = self.convert_timestamp_to_string(trial_data['recommended_song'])

        # Append to results list
        self.results.append(trial_data)

    def finalise_results(self):
        """
        Finalizes the results of the simulation by converting the list of trial results into a DataFrame. It calculates
        the accuracy of predictions and saves the DataFrame to a CSV file specified in the configuration.
        """
        # Convert to df
        self.results_df = pd.DataFrame(self.results)
        
        # Calculate scores column 
        self.results_df['score'] = self.results_df.apply(lambda row: 1 if row['predicted_action'] == row['actual_action'] else 0, axis=1)
        
        # Save to df
        self.results_df.to_csv(self.results_path, index=False)

    def log_text(self, text: str):
        """
        Appends a given piece of text to the simulation's text log file. This is typically used for detailed logging
        of each trial's prompts, responses, and outcomes.

        Args:
            text (str): The text to be logged, such as prompts, responses, and custom messages.
        """
        with open(self.text_log_path, 'a') as file:
            file.write(text + "\n")

    def save_config(self):
        """Saves the config dict to a json file at instantiation."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)

    def convert_timestamp_to_string(self, song: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts 'listen_timestamp' in a song dictionary from a Timestamp object to a string.
        
        Args:
            song (Song): The song dictionary containing possibly a 'listen_timestamp' key.
        
        Returns:
            Dict[str, Any]: The song dictionary with 'listen_timestamp' converted to string format if it exists.
        """
        if 'listen_timestamp' in song and isinstance(song['listen_timestamp'], pd.Timestamp):
            song['listen_timestamp'] = song['listen_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        return song

    @staticmethod
    def concatenate_dfs_with_unique_trial_ids(dfs_list: List[pd.DataFrame]):
        """
        Concatenates a list of DataFrames (dfs) while ensuring that trial_ids are unique across the combined DataFrame.
        
        Args:
            dfs_list (List[pd.DataFrame]): List of DataFrames to concatenate.
            
        Returns:
            pd.DataFrame: A single DataFrame with unique trial_ids.
        """
        max_trial_id = 0  # Initialize the maximum trial_id found so far
        
        for i, df in enumerate(dfs_list):
            if i > 0:  # Skip this step for the first DataFrame
                # Calculate the adjustment needed for the current DataFrame
                adjustment = max_trial_id + 1
                df['trial_id'] = df['trial_id'] + adjustment
            
            # Update the max_trial_id for the next iteration
            current_max_trial_id = df['trial_id'].max()
            max_trial_id = current_max_trial_id
        
        # Concatenate all the adjusted DataFrames
        concatenated_df = pd.concat(dfs_list, ignore_index=True)
        
        return concatenated_df
        
    @staticmethod
    def calculate_p_value(successes, trials):
        """Calculates the p-value for a given number of successes and trials using a binomial test with a success probability of 0.5.

        Args:
            successes (float or int): The number of successes observed in the experiment. This value is rounded to the nearest integer.
            trials (int): The total number of trials conducted in the experiment.
            
        Returns:
            float: The p-value from the binomial test, indicating the probability of observing the given number of successes (or more 
                    extreme) under the null hypothesis of equal chance of success and failure (p=0.5).
        """
        # Ensure inputs are integers
        successes = round(successes)
        trials = int(trials)
        return binomtest(successes, trials, p=0.5).pvalue
        
    @staticmethod
    def print_hyperparameter_comparison_results(results_df: pd.DataFrame,
                                                num_recommendations: int = 3,
                                                length_history_sequence: int = 5):
        """
        Analyzes and prints the comparison results of different simulation settings, including user gender, song recommendation position, lyric options, and user profile options.
        
        This method calculates the overall accuracy and performs a binomial test to compare these scores against random chance. It then iterates through specified hyperparameters to compare their impact on recommendation accuracy, finally presenting the best hyperparameter combinations based on average scores.
        
        Args:
            results_df (pd.DataFrame): The DataFrame containing simulation results to analyze.
            num_recommendations (int, optional): The number of song recommendations made. Defaults to 3.
            length_history_sequence (int, optional): The length of the user's song history sequence considered. Defaults to 5.
        """
        # Get num_trials etc. from dataframe 
        num_trials = results_df['trial_id'].max() + 1
        # num_recommendations = results_df['idx_recommended_song'].max() + 1
        #length_history_sequence = len(results_df['song_history'].iloc[0])
        # The last line of code there does not work because of interpreting the literals. There is an issue 
        # with interpreting the Timestamp. 

        # Calculate overall accuracy and perform binomial test
        overall_accuracy = results_df['score'].mean()
        overall_successes = results_df['score'].sum()
        overall_trials = len(results_df)
        p_value_overall = binomtest(overall_successes, overall_trials, p=0.5).pvalue

        # Print overall results
        print(f"total_trials = {overall_trials}; num_trials: {num_trials}; num_recommendations: {num_recommendations}, length_history_sequence: {length_history_sequence}")
        print(f"\nOverall Accuracy: {overall_accuracy:.2f}")
        print(f"P-value (Overall vs Random Chance): {p_value_overall:.4f}")
        if p_value_overall < 0.05:
            print("  ChatGPT's overall performance is statistically significantly different from random chance (p<0.05).")
        else:
            print("  ChatGPT's overall performance is not statistically significantly different from random chance (p>=0.05).")

        # Define hyperparameters to compare
        hyperparameters = ['user_gender', 'idx_recommended_song', 'lyric_option', 'user_profile_option']

        # Compare each hyperparameter
        for param in hyperparameters:
            print(f"\nComparing '{param}' Accuracy:")
            groups = results_df.groupby(param)

            # For each group, print the average score and perform binomial test
            for name, group in groups:
                avg_score = group['score'].mean()
                successes = group['score'].sum()
                trials = len(group)
                p_value = binomtest(successes, trials, p=0.5).pvalue

                print(f"  {name}: {avg_score:.2f}, P-value: {p_value:.4f} (n={trials})")

        # Begin new code for analyzing best hyperparameter combinations
        # Remove 'user_gender' and 'idx_recommended_song' as they're not _really_ hyperparameters
        hyperparameters.remove('user_gender')
        hyperparameters.remove('idx_recommended_song')
        print("\nBest Hyperparameter Combinations (lyric_option-user_profile_option):")
        results_df['combination'] = results_df[hyperparameters].astype(str).agg('-'.join, axis=1)
        
        # Initialize an empty list to collect data
        agg_results_data = []

        for combination, group in results_df.groupby('combination'):
            avg_score = group['score'].mean()
            successes = round(group['score'].sum())  # Total successes is the sum of scores, rounded
            trials = len(group)  # Number of trials is the count of rows in the group
            p_value = SimulationManager.calculate_p_value(successes, trials)

            # Append a dictionary with the results for this combination to the list
            agg_results_data.append({
                'Combination': combination,
                'AvgScore': avg_score,
                'Trials': trials,
                'PValue': p_value
            })

        # Convert the list of dictionaries to a DataFrame all at once
        agg_results = pd.DataFrame(agg_results_data)

        # Sort the aggregated results by AvgScore in descending order
        agg_results = agg_results.sort_values(by='AvgScore', ascending=False)

        # Print the DataFrame as a table
        print(agg_results.to_string(index=False))