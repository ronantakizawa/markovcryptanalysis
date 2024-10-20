import numpy as np
import random
import string
import math
from collections import Counter, defaultdict
import nltk
import logging
import time
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

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

# AES Encryption
def encrypt_aes(plaintext, key):
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.iv
    ciphertext = cipher.encrypt(pad(plaintext.encode('utf-8'), AES.block_size))
    return iv + ciphertext  # Prepend the IV for decryption

# AES Decryption
def decrypt_aes(ciphertext, key):
    iv = ciphertext[:AES.block_size]  # Extract the IV from the start
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext[AES.block_size:]), AES.block_size)
    return plaintext.decode('utf-8')

# Generate a random AES key
def generate_aes_key():
    return get_random_bytes(16)  # 16 bytes = 128 bits

def combined_score_text(text, models, word_list):
    total_score = 0.0
    for n, model in models.items():
        ngram_score = score_text(text, model, n)
        if n == 1:
            weight = 0.2  # Lower weight for unigrams
        elif n == 2:
            weight = 1.0  # Weight bigrams more
        elif n == 3:
            weight = 2.0  # Weight trigrams heavily
        else:
            weight = 0.5  # Lower weights for higher-order n-grams
        total_score += weight * ngram_score
    # Add word count to the score
    decrypted_words = text.split()
    valid_word_count = sum(1 for word in decrypted_words if word in word_list)
    total_score += valid_word_count * 15  # Heavier weight on valid words
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

def simulated_annealing_aes(ciphertext, models, word_list, key, max_iterations=10000, temperature=100.0, cooling_rate=0.0001):
    # Start timer
    start_time = time.time()
    
    iv = ciphertext[:AES.block_size]  # Extract the IV from the ciphertext
    current_plaintext = decrypt_aes(ciphertext, key)  # Decrypt the ciphertext with the initial key
    current_score = combined_score_text(current_plaintext, models, word_list)
    
    best_key = key
    best_score = current_score
    
    iterations = 0  # Initialize iteration counter
    
    # Data collection for plotting
    scores_over_time = []
    
    for i in range(max_iterations):
        iterations += 1
        # Decrease the temperature
        temp = temperature / (1 + cooling_rate * i)
        
        # Generate a new random AES key (mutate the key)
        new_key = generate_aes_key()  # The new key for this iteration
        
        try:
            new_plaintext = decrypt_aes(ciphertext, new_key)  # Decrypt the ciphertext with the new key
        except ValueError:
            # If decryption fails due to padding error, continue without updating the key
            continue
        
        new_score = combined_score_text(new_plaintext, models, word_list)
        
        # Decide whether to accept the new key
        delta_score = new_score - current_score
        if delta_score > 0 or math.exp(delta_score / temp) > random.random():
            key = new_key
            current_score = new_score
            current_plaintext = new_plaintext
            
            if current_score > best_score:
                best_key = key
                best_score = current_score
                logging.info(f"Iteration {i}, Best Score: {best_score}")
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
        
        # Encrypt the plaintext using AES
        key = generate_aes_key()
        ciphertext = encrypt_aes(sample_plaintext, key)
        logging.info(f"AES Ciphertext (Hex): {ciphertext.hex()}")

        # Decrypt the ciphertext
        decrypted_text = decrypt_aes(ciphertext, key)
        logging.info(f"Decrypted text: {decrypted_text}")
        
        # Log the performance
        logging.info("\nOriginal vs Decrypted:")
        for orig_char, dec_char in zip(sample_plaintext, decrypted_text):
            logging.info(f"{orig_char} -> {dec_char}")
        
        # Perform simulated annealing on the AES encrypted text
        recovered_key, score, iterations, time_taken, scores_over_time = simulated_annealing_aes(
            ciphertext, models, word_list, key, max_iterations=10000, temperature=100.0, cooling_rate=0.0001
        )
        
        logging.info(f"Recovered key: {recovered_key.hex()}")
        
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
