### Chapter 2 Exercise 5, continuation
### noting that KenLM has some deprecations, thus had to do some manual installs of the library on my local
"""
Train a KenLM model on your own dataset. Use a portion of the ‘realnewslike’ subset of C4 and train the 
model using the instructions provided in the KenLM Github page. 
You can then calculate the perplexity of each document in a subset of the unclean version of C4. 
"""

import os  
import subprocess  # Importing subprocess to run shell commands from Python
import json  

def convert_json_to_text(json_file, text_file):
    """
    Converts a JSON file containing text data into a plain text file.
    Each line in the JSON file is expected to be a JSON object with a 'text' field.
    This is to processes the C4 realnewslike dataset file, which I downloaded from https://huggingface.co/datasets/allenai/c4/blob/main/realnewslike/c4-train.00138-of-00512.json.gz
    """
    print(f"Converting {json_file} to {text_file}...")
    with open(json_file, 'r') as f:  # Open the JSON file for reading
        with open(text_file, 'w') as out:  # Open the text file for writing
            for line in f:
                data = json.loads(line)  # Parse each line as a JSON object
                out.write(data['text'] + '\n')  # Write the 'text' field to the text file
    print("Conversion complete.")

def build_kenlm_model(text_file, arpa_file, order=5):
    """
    Builds a KenLM language model from a text file and saves it as an ARPA file.
    The 'order' parameter specifies the n-gram order of the model.
    Double check your local to make sure there's an ARPA file, which is the actual trained model.
    Because the KenLM library wasn't fully working on Google Colab, I ran these commands on terminal instead, based on the documentation. 
    Note that I did some manual installs to ensure that the bin folder was correctly extracted. 
    """
    print(f"Building KenLM model from {text_file} to {arpa_file}...")
    subprocess.run(['kenlm/build/bin/lmplz', '-o', str(order), '--text', text_file, '--arpa', arpa_file], check=True)
    # Run the KenLM lmplz command to build the model
    print("Model building complete.")

def calculate_perplexity(arpa_file, text_file):
    """
    Calculates the perplexity of a text file using a pre-built KenLM ARPA model.
    """
    print(f"Calculating perplexity for {text_file} using {arpa_file}...")
    result = subprocess.run(['kenlm/build/bin/query', arpa_file], input=open(text_file).read(), capture_output=True, text=True)
    # Run the KenLM query command to calculate perplexity
    print("Perplexity calculation complete.")
    print("Perplexity results:")  # Print the perplexity results
    # Extract and print only the important results (I printed all the perplexities manually earlier, but my OS ran out of storage, so now just printing key scores)
    important_lines = [line for line in result.stdout.splitlines() if "Perplexity" in line or "OOVs" in line or "Tokens" in line]
    for line in important_lines:
        print(line) # print in terminal to double check
    with open('perplexity_results.txt', 'w') as f:
        for line in important_lines:
            f.write(line + '\n')  # Save to file

def main():
    json_file = 'c4-train.00138-of-00512.json'  # Input JSON file
    text_file = 'c4-train.00138-of-00512.txt'  # Intermediate text file; convert to txt for model processing
    arpa_file = 'c4-train.00138-of-00512.arpa'  # Output ARPA file, model file
    
    # Convert JSON to text
    # This step prepares the data for model training
    convert_json_to_text(json_file, text_file)
    
    # Build KenLM model
    # This step creates the language model from the text data
    build_kenlm_model(text_file, arpa_file)
    
    # Calculate perplexity
    # This step evaluates the model's performance on the text data
    calculate_perplexity(arpa_file, text_file)

if __name__ == '__main__':
    main()
