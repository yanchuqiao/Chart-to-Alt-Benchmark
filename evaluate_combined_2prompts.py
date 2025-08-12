import json
import evaluate
from sentence_transformers import SentenceTransformer, util

# Load the ground truth data
with open('/home/ubuntu/Chart-to_Alt/test.json', 'r') as f:
    ground_truth_data = json.load(f)

# Build combined ground truth dict
ground_truth_combined = {}
for example in ground_truth_data:
    img_id = example['image_id']
    caption_combined = example['caption_L1'].strip() + " " + example['caption_L2L3'].strip()
    ground_truth_combined[img_id] = caption_combined

# Load the generated summaries
generated_combined = {}
with open('/home/ubuntu/Chart-to_Alt/third_finetuning_inference_m2_v2.txt', 'r') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 2):
        if lines[i].startswith("Generated text for image"):
            image_filename = lines[i].split(" ")[4].split(":")[0]
            l1 = lines[i].split("(caption_L1): ")[1].strip()
            l2l3 = lines[i+1].split("(caption_L2L3): ")[1].strip()
            generated_combined[image_filename] = l1 + " " + l2l3

# Load models and metrics
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
bleu = evaluate.load("bleu")
ter = evaluate.load("ter")
rouge = evaluate.load("rouge")

def compute_cosine_similarity(text1, text2):
    embedding_1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding_2 = sentence_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2).item()

# Evaluation loop
bleu_scores = []
rougeL_scores = []
ter_scores = []
cosine_sim_scores = []

with open('/home/ubuntu/Chart-to_Alt/Tinychart_combined_score_m2_v2.txt', 'w') as output_file:
    for image_id, gen_caption in generated_combined.items():
        gt_caption = ground_truth_combined.get(image_id)
        if not gt_caption:
            continue

        bleu_score = bleu.compute(predictions=[gen_caption], references=[gt_caption])['bleu']
        rougeL_score = rouge.compute(predictions=[gen_caption], references=[gt_caption])['rougeL']
        ter_score = ter.compute(predictions=[gen_caption], references=[gt_caption])['score']
        cosine_score = compute_cosine_similarity(gen_caption, gt_caption)

        bleu_scores.append(bleu_score)
        rougeL_scores.append(rougeL_score)
        ter_scores.append(ter_score)
        cosine_sim_scores.append(cosine_score)

        output_file.write(f"Image: {image_id}\n")
        output_file.write(f"Ground Truth: {gt_caption}\n")
        output_file.write(f"Generated: {gen_caption}\n")
        output_file.write(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rougeL_score:.4f}, TER: {ter_score:.4f}, CosineSim: {cosine_score:.4f}\n\n")

    # Averages
    output_file.write(f"\nAverage BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}\n")
    output_file.write(f"Average ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}\n")
    output_file.write(f"Average TER: {sum(ter_scores)/len(ter_scores):.4f}\n")
    output_file.write(f"Average Cosine Similarity: {sum(cosine_sim_scores)/len(cosine_sim_scores):.4f}\n")
