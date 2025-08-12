import json
import evaluate
from sentence_transformers import SentenceTransformer, util
import pdb
# from nltk.corpus import stopwords
# from nltk import download
# import gensim.downloader as api

# download('stopwords')
# stop_words = stopwords.words('english')

# Load the ground truth data from the JSON file
with open('/home/ubuntu/Chart-to_Alt/test.json', 'r') as f:
    ground_truth_data = json.load(f)

# Create dictionaries to store ground truth summaries indexed by image filenames for L1 and L2L3 captions
ground_truth_dict_L1 = {}
ground_truth_dict_L2L3 = {}

for example in ground_truth_data:
    img_id = example['image_id']
    caption_L1 = example['caption_L1']
    caption_L2L3 = example['caption_L2L3']

    if img_id not in ground_truth_dict_L1:
        ground_truth_dict_L1[img_id] = []
    ground_truth_dict_L1[img_id].append(caption_L1)

    if img_id not in ground_truth_dict_L2L3:
        ground_truth_dict_L2L3[img_id] = []
    ground_truth_dict_L2L3[img_id].append(caption_L2L3)

# Load the generated summaries from the output file
generated_summaries_L1 = {}
generated_summaries_L2L3 = {}
with open('/home/ubuntu/Chart-to_Alt/third_finetuning_inference_m2_v2.txt', 'r') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 2):  # Skip every other line
        if lines[i].startswith("Generated text for image"):
            image_filename = lines[i].split(" ")[4].split(":")[0]#.split('.')[0]
            generated_L1_summary = lines[i].split("(caption_L1): ")[1].strip()
            generated_L2L3_summary = lines[i+1].split("(caption_L2L3): ")[1].strip()
            generated_summaries_L1.setdefault(image_filename, []).append(generated_L1_summary)
            generated_summaries_L2L3.setdefault(image_filename, []).append(generated_L2L3_summary)

# Initialize lists to store BLEU, ROUGE, and BLEURT scores for L1 and L2L3 captions
bleu_scores_L1 = []
bleu_scores_L2L3 = []
rougeL_scores_L1 =[]
rougeL_scores_L2L3 = []
ter_scores_L1 = []
ter_scores_L2L3 = []
cosine_sim_scores_L1 = []
cosine_sim_scores_L2L3 = []
# wmd_scores_L1 = []
# wmd_scores_L2L3 = []

# Load models
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
bleu = evaluate.load("bleu")
ter = evaluate.load("ter")
rouge = evaluate.load("rouge")
#model_w2v = api.load('word2vec-google-news-300')

# Preprocessing for WMD
# def preprocess(text):
#     return [w for w in text.lower().split() if w not in stop_words]

def compute_cosine_similarity(text1, text2):
    embedding_1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding_2 = sentence_model.encode(text2, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embedding_1, embedding_2)
    return cosine_sim.item()

# Open a text file to save scores and summaries
with open('/home/ubuntu/Chart-to_Alt/Tinychart_seperate_score_m2_v2.txt', 'w') as output_file:
    # Set to track evaluated pairs
    evaluated_pairs = set()

    # Iterate over images and calculate BLEU, ROUGE, and BLEURT scores for L1 captions
    for image_filename, generated_L1_summaries in generated_summaries_L1.items():
        ground_truth_L1_summaries = ground_truth_dict_L1.get(image_filename, [])
        if ground_truth_L1_summaries:
            for generated_L1_summary in generated_L1_summaries:
                # Calculate BLEU, ROUGE, and BLEURT scores for each ground truth caption
                for ground_truth_L1_summary in ground_truth_L1_summaries:
                    pair = (image_filename, generated_L1_summary, ground_truth_L1_summary)
                    if pair not in evaluated_pairs:
                        evaluated_pairs.add(pair)
                        # Calculate BLEU score for L1 captions
                        bleu_score_L1 = bleu.compute(predictions=[generated_L1_summary], references=[ground_truth_L1_summary])
                        bleu_scores_L1.append(bleu_score_L1['bleu'])
                        # Calculate ROUGE-L score for L1 captions
                        rouge_score_L1 = rouge.compute(predictions=[generated_L1_summary], references=[ground_truth_L1_summary])
                        rougeL_scores_L1.append(rouge_score_L1['rougeL'])
                        # Calculate TER score for L1 captions
                        ter_score_L1 = ter.compute(predictions=[generated_L1_summary], references=[ground_truth_L1_summary])
                        ter_scores_L1.append(ter_score_L1['score'])
                        # Calculate Cosine Similarity score for L1 captions
                        cosine_sim_score_L1 = compute_cosine_similarity(generated_L1_summary, ground_truth_L1_summary)
                        cosine_sim_scores_L1.append(cosine_sim_score_L1)
                        # Calculate WMD score for L1 captions
                        # try:
                        #     wmd_score = model_w2v.wmdistance(preprocess(generated_L1_summary), preprocess(ground_truth_L1_summary))
                        # except:
                        #     wmd_score = float('inf')
                        # wmd_scores_L1.append(wmd_score)
                        # Write to output file
                        output_file.write(f"Image: {image_filename}\n")
                        output_file.write(f"Ground Truth L1 Summary: {ground_truth_L1_summary}\n")
                        output_file.write(f"Generated L1 Summary: {generated_L1_summary}\n")
                        output_file.write(f"BLEU Score for L1: {bleu_score_L1}\n")
                        output_file.write(f"ROUGE-L Score for L1: {rouge_score_L1['rougeL']}\n")
                        output_file.write(f"TER Score for L1: {ter_score_L1}\n")
                        output_file.write(f"Cosine Similarity for L1: {cosine_sim_score_L1}\n")
                        #output_file.write(f"WMD for L1: {wmd_scores_L1}\n")
                        output_file.write("\n")

    # Clear the set for L2L3 evaluation
    evaluated_pairs.clear()

    # Iterate over images and calculate BLEU, ROUGE, and BLEURT scores for L2L3 captions
    for image_filename, generated_L2L3_summaries in generated_summaries_L2L3.items():
        ground_truth_L2L3_summaries = ground_truth_dict_L2L3.get(image_filename, [])
        if ground_truth_L2L3_summaries:
            for generated_L2L3_summary in generated_L2L3_summaries:
                # Calculate BLEU, ROUGE, and BLEURT scores for each ground truth caption
                for ground_truth_L2L3_summary in ground_truth_L2L3_summaries:
                    pair = (image_filename, generated_L2L3_summary, ground_truth_L2L3_summary)
                    if pair not in evaluated_pairs:
                        evaluated_pairs.add(pair)
                        # Calculate BLEU score for L2L3 captions
                        bleu_score_L2L3 = bleu.compute(predictions=[generated_L2L3_summary], references=[ground_truth_L2L3_summary])
                        bleu_scores_L2L3.append(bleu_score_L2L3['bleu'])
                        # Calculate ROUGE-L score for L2L3 captions
                        rouge_score_L2L3 = rouge.compute(predictions=[generated_L2L3_summary], references=[ground_truth_L2L3_summary])
                        rougeL_scores_L2L3.append(rouge_score_L2L3['rougeL'])
                        # Calculate TER score for L2L3 captions
                        ter_score_L2L3 = ter.compute(predictions=[generated_L2L3_summary], references=[ground_truth_L2L3_summary])
                        ter_scores_L2L3.append(ter_score_L2L3['score'])
                        # Calculate Cosine Similarity score for L2L3 captions
                        cosine_sim_score_L2L3 = compute_cosine_similarity(generated_L2L3_summary, ground_truth_L2L3_summary)
                        cosine_sim_scores_L2L3.append(cosine_sim_score_L2L3)
                        # Calculate WMD score for L1 captions
                        # try:
                        #     wmd_score = model_w2v.wmdistance(preprocess(generated_L2L3_summary), preprocess(ground_truth_L2L3_summary))
                        # except:
                        #     wmd_score = float('inf')
                        # wmd_scores_L2L3.append(wmd_score)
                        # Write to output file
                        output_file.write(f"Image: {image_filename}\n")
                        output_file.write(f"Ground Truth L2L3 Summary: {ground_truth_L2L3_summary}\n")
                        output_file.write(f"Generated L2L3 Summary: {generated_L2L3_summary}\n")
                        output_file.write(f"BLEU Score for L2L3: {bleu_score_L2L3}\n")
                        output_file.write(f"ROUGE-L Score for L2L3: {rouge_score_L2L3['rougeL']}\n")
                        output_file.write(f"TER Score for L2L3: {ter_score_L2L3}\n")
                        output_file.write(f"Cosine Similarity for L2L3: {cosine_sim_score_L2L3}\n")
                        #output_file.write(f"WMD for L2L3: {wmd_scores_L2L3}\n")
                        output_file.write("\n")

    # Calculate average BLEU, ROUGE, and BLEURT scores for L1 and L2L3 captions
    if bleu_scores_L1:
        avg_bleu_score_L1 = sum(bleu_scores_L1) / len(bleu_scores_L1)
        print("Average BLEU Score for L1 captions:", avg_bleu_score_L1)
        output_file.write(f"Average BLEU Score for L1 captions: {avg_bleu_score_L1}\n")
    else:
        print("No matched pairs found for L1 captions.")
        output_file.write("No matched pairs found for L1 captions.\n")

    if bleu_scores_L2L3:
        avg_bleu_score_L2L3 = sum(bleu_scores_L2L3) / len(bleu_scores_L2L3)
        print("Average BLEU Score for L2L3 captions:", avg_bleu_score_L2L3)
        output_file.write(f"Average BLEU Score for L2L3 captions: {avg_bleu_score_L2L3}\n")
    else:
        print("No matched pairs found for L2L3 captions.")
        output_file.write("No matched pairs found for L2L3 captions.\n")

    if rougeL_scores_L1:
        avg_rougeL_score_L1 = sum(rougeL_scores_L1) / len(rougeL_scores_L1)
        print("Average ROUGE-L Score for L1 captions:", avg_rougeL_score_L1)
        output_file.write(f"Average ROUGE-L Score for L1 captions: {avg_rougeL_score_L1}\n")
    else:
        print("No matched pairs found for L1 captions.")
        output_file.write("No matched pairs found for L1 captions.\n")

    if rougeL_scores_L2L3:
        avg_rougeL_score_L2L3 = sum(rougeL_scores_L2L3) / len(rougeL_scores_L2L3)
        print("Average ROUGE-L Score for L2L3 captions:", avg_rougeL_score_L2L3)
        output_file.write(f"Average ROUGE-L Score for L2L3 captions: {avg_rougeL_score_L2L3}\n")
    else:
        print("No matched pairs found for L2L3 captions.")
        output_file.write("No matched pairs found for L2L3 captions.\n")

    if ter_scores_L1:
        avg_ter_score_L1 = sum(ter_scores_L1) / len(ter_scores_L1)
        print("Average TER Score for L1 captions:", avg_ter_score_L1)
        output_file.write(f"Average TER Score for L1 captions: {avg_ter_score_L1}\n")
    else:
        print("No matched pairs found for L1 captions.")
        output_file.write("No matched pairs found for L1 captions.\n")

    if ter_scores_L2L3:
        avg_ter_score_L2L3 = sum(ter_scores_L2L3) / len(ter_scores_L2L3)
        print("Average TER Score for L2L3 captions:", avg_ter_score_L2L3)
        output_file.write(f"Average TER Score for L2L3 captions: {avg_ter_score_L2L3}\n")
    else:
        print("No matched pairs found for L2L3 captions.")
        output_file.write("No matched pairs found for L2L3 captions.\n")

    if cosine_sim_scores_L1:
        avg_cosine_sim_score_L1 = sum(cosine_sim_scores_L1) / len(cosine_sim_scores_L1)
        print("Average Cosine Similarity for L1 captions:", avg_cosine_sim_score_L1)
        output_file.write(f"Average Cosine Similarity for L1 captions: {avg_cosine_sim_score_L1}\n")
    else:
        print("No matched pairs found for L1 captions.")
        output_file.write("No matched pairs found for L1 captions.\n")

    if cosine_sim_scores_L2L3:
        avg_cosine_sim_score_L2L3 = sum(cosine_sim_scores_L2L3) / len(cosine_sim_scores_L2L3)
        print("Average Cosine Similarity for L2L3 captions:", avg_cosine_sim_score_L2L3)
        output_file.write(f"Average Cosine Similarity for L2L3 captions: {avg_cosine_sim_score_L2L3}\n")
    else:
        print("No matched pairs found for L2L3 captions.")
        output_file.write("No matched pairs found for L2L3 captions.\n")

    # if wmd_scores_L1:
    #     avg_wmd_score_L1 = sum(wmd_scores_L1) / len(wmd_scores_L1)
    #     print("Average WMD for L1 captions:", avg_wmd_score_L1)
    #     output_file.write(f"Average WMD for L1 captions: {avg_wmd_score_L1}\n")
    # else:
    #     print("No matched pairs found for L1 captions.")
    #     output_file.write("No matched pairs found for L1 captions.\n")

    # if wmd_scores_L2L3:
    #     avg_wmd_score_L2L3 = sum(wmd_scores_L2L3) / len(wmd_scores_L2L3)
    #     print("Average WMD for L1 captions:", avg_wmd_score_L2L3)
    #     output_file.write(f"Average WMD for L1 captions: {avg_wmd_score_L2L3}\n")
    # else:
    #     print("No matched pairs found for L2L3 captions.")
    #     output_file.write("No matched pairs found for L2L3 captions.\n")