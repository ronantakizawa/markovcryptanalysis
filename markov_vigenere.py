import numpy as np
import random
import string
import math
from collections import Counter, defaultdict
import nltk
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data files (run once)
nltk.download('words')

from nltk.corpus import words

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def preprocess_text(file_paths):
    combined_text = ''
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Preprocess each text
            text = text.lower()
            text = ''.join(filter(lambda c: c in string.ascii_lowercase + ' ', text))
            text = ' '.join(text.split())
            combined_text += ' ' + text
    return combined_text

def build_markov_model(text, n=3):
    model = defaultdict(Counter)
    for i in range(len(text) - n):
        ngram = text[i:i+n]
        next_char = text[i + n]
        model[ngram][next_char] += 1
    # Convert counts to probabilities
    for ngram, counter in model.items():
        total = sum(counter.values())
        for char in counter:
            counter[char] /= total
    return model

def generate_keyword(length):
    letters = string.ascii_lowercase
    keyword = ''.join(random.choice(letters) for _ in range(length))
    return keyword

def encrypt_vigenere(plaintext, keyword):
    ciphertext = ''
    keyword_length = len(keyword)
    letters = string.ascii_lowercase
    for i, char in enumerate(plaintext):
        if char in letters:
            shift = letters.index(keyword[i % keyword_length])
            encrypted_char = letters[(letters.index(char) + shift) % 26]
            ciphertext += encrypted_char
        else:
            ciphertext += char
    return ciphertext

def decrypt_vigenere(ciphertext, keyword):
    plaintext = ''
    keyword_length = len(keyword)
    letters = string.ascii_lowercase
    for i, char in enumerate(ciphertext):
        if char in letters:
            shift = letters.index(keyword[i % keyword_length])
            decrypted_char = letters[(letters.index(char) - shift) % 26]
            plaintext += decrypted_char
        else:
            plaintext += char
    return plaintext

def kasiski_examination(ciphertext, max_key_length=20):
    # Find repeated sequences of length 3 or more
    repeated_sequences = {}
    for seq_len in range(3, 6):  # Check sequences of length 3 to 5
        for i in range(len(ciphertext) - seq_len):
            seq = ciphertext[i:i+seq_len]
            for j in range(i+seq_len, len(ciphertext) - seq_len):
                if ciphertext[j:j+seq_len] == seq:
                    distance = j - i
                    repeated_sequences.setdefault(seq, []).append(distance)
    # Find factors of distances
    factors = Counter()
    for distances in repeated_sequences.values():
        for distance in distances:
            for factor in range(2, max_key_length + 1):
                if distance % factor == 0:
                    factors[factor] += 1
    # Most likely key lengths are the most common factors
    likely_key_lengths = [pair[0] for pair in factors.most_common()]
    return likely_key_lengths

def frequency_analysis_on_segments(ciphertext, key_length):
    letters = string.ascii_lowercase
    key = ''
    for i in range(key_length):
        # Extract the segment and filter out non-letter characters
        segment = ''.join(c for c in ciphertext[i::key_length] if c in letters)
        if not segment:
            # If the segment is empty, default to 'a' (no shift)
            key += 'a'
            continue
        freq_counter = Counter(segment)
        # Assume 'e' is the most common letter in English
        most_common_char = freq_counter.most_common(1)[0][0]
        shift = (letters.index(most_common_char) - letters.index('e')) % 26
        key_letter = letters[shift]
        key += key_letter
    return key

def combined_score_text(text, models, word_list):
    total_score = 0.0
    for n, model in models.items():
        ngram_score = score_text(text, model, n)
        if n == 1:
            weight = 0.5  # Lower weight for unigrams
        else:
            weight = n  # Weight higher-order n-grams more heavily
        total_score += weight * ngram_score
    # Add word count to the score
    decrypted_words = text.split()
    valid_word_count = sum(1 for word in decrypted_words if word in word_list)
    total_score += valid_word_count * 10  # Increase the weight
    return total_score

def score_text(text, model, n=3):
    score = 0.0
    for i in range(len(text) - n):
        ngram = text[i:i+n]
        next_char = text[i + n]
        if ngram in model and next_char in model[ngram]:
            score += math.log(model[ngram][next_char])
        else:
            # Assign a low probability to unseen n-grams
            score += math.log(1e-6)
    return score

def simulated_annealing_vigenere(ciphertext, models, word_list, key_length, max_iterations=10000, temperature=100.0, cooling_rate=0.0001):
    # Start timer
    start_time = time.time()
    
    letters = string.ascii_lowercase
    # Initialize key randomly
    current_key = generate_keyword(key_length)
    current_plaintext = decrypt_vigenere(ciphertext, current_key)
    current_score = combined_score_text(current_plaintext, models, word_list)
    
    best_key = current_key
    best_score = current_score
    
    iterations = 0  # Initialize iteration counter
    
    # Data collection for plotting
    scores_over_time = []
    
    for i in range(max_iterations):
        iterations += 1
        # Decrease the temperature
        temp = temperature / (1 + cooling_rate * i)
        # Mutate the key: change one character
        new_key = list(current_key)
        pos = random.randint(0, key_length -1)
        new_char = random.choice(letters)
        new_key[pos] = new_char
        new_key = ''.join(new_key)
        
        new_plaintext = decrypt_vigenere(ciphertext, new_key)
        new_score = combined_score_text(new_plaintext, models, word_list)
        
        # Decide whether to accept the new key
        delta_score = new_score - current_score
        if delta_score > 0 or math.exp(delta_score / temp) > random.random():
            current_key = new_key
            current_score = new_score
            current_plaintext = new_plaintext
            
            if current_score > best_score:
                best_key = current_key
                best_score = current_score
                logging.info(f"Iteration {i}, Best Score: {best_score}")
                logging.info(f"Current Key: {best_key}")
                logging.info("Current Decryption:")
                logging.info(current_plaintext[:100])
                logging.info("-" * 50)
        # Collect scores for plotting
        scores_over_time.append(best_score)
    # End timer
    end_time = time.time()
    time_taken = end_time - start_time
    return best_key, best_score, iterations, time_taken, scores_over_time

def main():
    # Step 1: Preprocess the text corpus
    file_paths = [
        'pride_and_prejudice.txt',
        'sense_and_sensibility.txt',
        'moby_dick.txt',
        'great_expectations.txt',
        'dracula.txt',
        'war_and_peace.txt',
    ]
    corpus_text = preprocess_text(file_paths)
    logging.info("Text corpus processed.")
    
    # Step 2: Build the Markov models for different n-grams
    n_values = [1, 2, 3, 4]  # Include unigrams
    models = {n: build_markov_model(corpus_text, n) for n in n_values}
    logging.info("Markov models built.")
    
    # Load word list
    word_list = set(words.words())
    logging.info("Word list loaded.")
    
    # Define multiple sample plaintexts
    sample_plaintexts = [
        "it is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife",
        "the quick brown fox jumps over the lazy dog",
        "to be or not to be that is the question",
        "all happy families are alike every unhappy family is unhappy in its own way",
        "call me ishmael some years ago never mind how long precisely having little or no money in my purse"
    ]
    
    # Initialize lists for plotting
    trial_numbers = []
    time_taken_list = []
    iterations_list = []
    
    plt.figure(figsize=(12, 8))
    
    # Loop over the sample plaintexts
    for idx, sample_plaintext in enumerate(sample_plaintexts, 1):
        logging.info(f"\n--- Trial {idx} ---")
        logging.info("Sample plaintext:")
        logging.info(sample_plaintext)
        
        # Generate a random keyword and encrypt the plaintext
        keyword_length = 5  # You can adjust the keyword length
        keyword = generate_keyword(keyword_length)
        ciphertext = encrypt_vigenere(sample_plaintext, keyword)
        logging.info(f"Keyword used for encryption (unknown to cryptanalysis): {keyword}")
        logging.info("Ciphertext:")
        logging.info(ciphertext)
        
        # Guess likely key lengths using Kasiski examination
        likely_key_lengths = kasiski_examination(ciphertext)
        logging.info(f"Likely key lengths: {likely_key_lengths}")
        
        # For simplicity, we'll use the first likely key length
        if likely_key_lengths:
            key_length = likely_key_lengths[0]
        else:
            key_length = 5  # Default to 5 if no likely key length found
        
        # Perform initial frequency analysis on segments to get starting key
        initial_key = frequency_analysis_on_segments(ciphertext, key_length)
        logging.info(f"Initial key guess based on frequency analysis: {initial_key}")
        
        # Step 4: Perform cryptanalysis using simulated annealing
        logging.info("Starting cryptanalysis...")
        recovered_key, score, iterations, time_taken, scores_over_time = simulated_annealing_vigenere(
            ciphertext, models, word_list, key_length, max_iterations=10000, temperature=100.0, cooling_rate=0.0001
        )
        
        # Decrypt the ciphertext with the recovered key
        decrypted_text = decrypt_vigenere(ciphertext, recovered_key)
        logging.info(f"Recovered key: {recovered_key}")
        logging.info("Decrypted text:")
        logging.info(decrypted_text)
        
        # Compare the decrypted text with the original plaintext
        logging.info("\nOriginal vs Decrypted:")
        for orig_char, dec_char in zip(sample_plaintext, decrypted_text):
            logging.info(f"{orig_char} -> {dec_char}")
        
        # Log performance metrics
        logging.info(f"Trial {idx} completed in {iterations} iterations and {time_taken:.2f} seconds.")
        logging.info("\n" + "="*60 + "\n")
        
        # Collect data for plotting
        trial_numbers.append(idx)
        time_taken_list.append(time_taken)
        iterations_list.append(iterations)
        
        # Plot the score over iterations for this trial
        plt.plot(scores_over_time, label=f'Trial {idx}')
    
    # Finalize the score over iterations plot
    plt.title('Best Score vs Iterations for Each Trial')
    plt.xlabel('Iterations')
    plt.ylabel('Best Score')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot time taken and iterations per trial
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Time Taken (s)', color=color)
    ax1.bar(trial_numbers, time_taken_list, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(trial_numbers)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Iterations', color=color)
    ax2.plot(trial_numbers, iterations_list, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Time Taken and Iterations per Trial')
    plt.show()
    
if __name__ == "__main__":
    main()