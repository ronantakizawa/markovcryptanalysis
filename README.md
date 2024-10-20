Notes:
- Cryptanalysis is the art and science of analyzing and breaking cryptographic systems.
- It involves finding weaknesses in cryptographic algorithms, protocols, or implementations that can be exploited to decrypt data without the key
- In language modeling, Markov Chains can predict the probability of a letter or word based on the preceding one.

Step-by-Step Execution:

Preprocessing the Text Corpus:

Reads multiple classic literature files.
Cleans and combines them into a single corpus (corpus_text).
Building Markov Models:

Constructs unigram to 4-gram models (n_values = [1, 2, 3, 4]) using the corpus.
These models will help in evaluating the probability of n-grams in decrypted texts.
Loading Word List:

Loads a set of valid English words from the NLTK corpus to aid in scoring decrypted texts.
Defining Sample Plaintexts:

Contains five different plaintexts of varying lengths and complexities to test the cryptanalysis tool.
Initializing Plotting Parameters:

Sets up lists to collect performance metrics (trial_numbers, time_taken_list, iterations_list).
Initializes a Matplotlib figure for plotting score progression.
Processing Each Trial:

For Each Plaintext:
Logging: Announces the trial number and displays the plaintext.
Encryption:
Generates a random keyword of length 5 (keyword_length = 5).
Encrypts the plaintext using the Vigenère cipher.
Logs the used keyword and ciphertext.
Kasiski Examination:
Analyzes the ciphertext to guess likely key lengths.
Logs the probable key lengths.
Chooses the first probable key length or defaults to 5 if none found.
Frequency Analysis:
Performs frequency analysis on segments of the ciphertext corresponding to each key character position.
Logs the initial key guess based on frequency analysis.
Simulated Annealing Cryptanalysis:
Runs the simulated annealing algorithm to optimize the key.
Logs progress during the optimization (iterations, best scores, recovered keys).
Decryption:
Decrypts the ciphertext using the recovered key.
Logs the decrypted text.
Comparison:
Compares each character of the decrypted text with the original plaintext.
Logs the character-by-character comparison.


How the project was built:
- Use a large, representative text corpus in the target language (e.g., English), Break the text into individual characters or n-grams.
- Define States and Transitions: States =  Individual characters or sequences (bigrams, trigrams), Transitions: Probability of moving from one state to another.
-  Calculate Transition Probabilities: Frequency Counts: Count how often each state transition occurs in the corpus, Probability Estimation: Divide the frequency of each transition by the total number of transitions from the originating state
- Construct the Transition Matrix: A matrix where each cell (i,j) represents the probability of transitioning from state  i to state j.
-  Implement Cryptanalysis Algorithm:  For a given decrypted text, calculate the overall probability using the Markov model,  Use logarithms to prevent underflow in probability calculations.
- Kasiski Examination:  Estimates the likely lengths of the keyword used in the Vigenère cipher by analyzing repeated sequences in the ciphertext, Searches for repeating n-grams (3 to 5 characters) in the ciphertext, For each repeated sequence, computes the distance (number of characters) between the occurrences,  Determines the common factors of these distances, as the keyword length is likely a factor of these distances,  Returns a list of probable key lengths sorted by their frequency of occurrence.
-  Frequency Analysis: Performs frequency analysis on each segment of the ciphertext corresponding to each character position in the keyword to make an initial guess of the keyword: Segment Extraction: For each position in the keyword, extracts every key_length-th character from the ciphertext, Filtering: Removes non-alphabetic characters from the segment to avoid skewing frequency counts.Frequency Counting: Counts the frequency of each character in the segment.
Initial Key Guess: Assumes that the most frequent character in each segment corresponds to 'e' (the most common letter in English). Calculates the shift needed to map the most frequent ciphertext character to 'e', thereby guessing the corresponding keyword character.
- Combined Scoring Function: Evaluates the likelihood that a decrypted text is correct by combining scores from various n-gram models and the count of valid English words, N-gram Scoring: For each n-gram model (unigram to 4-gram), calculates the log-probability score of the decrypted text, Weighting: Applies higher weights to higher-order n-grams to emphasize longer sequences, which are more indicative of correct decryption, Word Count Scoring: Counts the number of valid English words in the decrypted text and adds a weighted score for each valid word, encouraging decryption results that form meaningful words, Usage: Serves as the objective function for the simulated annealing algorithm, guiding it towards more plausible decryption keys.
- Scoring Individual Text Segments: Computes the log-probability score of a text segment based on the provided n-gram model.
- Simulated Annealing for Vigenère Cipher: Purpose: Optimizes the decryption key using the simulated annealing algorithm to maximize the score of the decrypted text: Usage: Central to breaking the Vigenère cipher by iteratively refining the keyword to maximize the likelihood of the decrypted text being correct.