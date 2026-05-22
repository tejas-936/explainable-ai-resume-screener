import re
import torch
import torch.nn.functional as F

def generate_counterfactual(text, swap_type):
    """
    Utility function to construct counterfactual text by swapping names,
    gendered pronouns, or demographic indicators.
    """
    if swap_type == "Swap to Female Pronouns/Names":
        replacements = {
            r'\bHe\b': 'She', r'\bhe\b': 'she',
            r'\bHim\b': 'Her', r'\bhim\b': 'her',
            r'\bHis\b': 'Her', r'\bhis\b': 'her',
            r'\bHimself\b': 'Herself', r'\bhimself\b': 'herself',
            r'\bMr\b\.?': 'Ms', r'\bmr\b\.?': 'ms',
            r'\bJohn\b': 'Jane', r'\bDavid\b': 'Sarah', r'\bMichael\b': 'Emily'
        }
    elif swap_type == "Swap to Male Pronouns/Names":
        replacements = {
            r'\bShe\b': 'He', r'\bshe\b': 'he',
            r'\bHer\b': 'Him', r'\bher\b': 'him',
            r'\bHers\b': 'His', r'\bhers\b': 'his',
            r'\bHerself\b': 'Himself', r'\bherself\b': 'herself',
            r'\bMs\b\.?': 'Mr', r'\bms\b\.?': 'mr', r'\bMrs\b\.?': 'Mr', r'\bmrs\b\.?': 'mr',
            r'\bJane\b': 'John', r'\bSarah\b': 'David', r'\bEmily\b': 'Michael'
        }
    elif swap_type == "Swap to Minority Demographics":
        replacements = {
            r'\bJohn\b': 'Jamal', r'\bDavid\b': 'Mateo', r'\bSarah\b': 'Aisha',
            r'\bMichael\b': 'Arjun', r'\bEmily\b': 'Mei', r'\bwhite\b': 'minority'
        }
    else:
        return text

    cf_text = text
    for pattern, replacement in replacements.items():
        cf_text = re.sub(pattern, replacement, cf_text)
    return cf_text

class ExplainableRecruitmentSuite:
    """
    A diagnostic suite for interpreting the representations and fairness of the HTVAE.
    Provides methods for latent space traversals, sentence-level attribution via occlusion,
    and counterfactual bias audits.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def latent_traversal(self, z_base, dim_to_traverse, traversal_range=(-3.0, 3.0), steps=5):
        """
        Perturbs a base latent vector along a specified dimension to study its effect
        on decoding outputs.
        """
        self.model.eval()
        traversals = []
        step_size = (traversal_range[1] - traversal_range[0]) / (steps - 1)
        
        with torch.no_grad():
            for i in range(steps):
                z_perturbed = z_base.clone()
                z_perturbed[0, dim_to_traverse] = traversal_range[0] + (i * step_size)
                traversals.append(z_perturbed)
                
        return traversals

    def sentence_level_attribution(self, input_ids, attention_mask, sentence_mask):
        """
        Calculates attribution scores for sentences using occlusion.
        Indicates the importance of each sentence to the latent representation.
        """
        self.model.eval()
        B, S, W = input_ids.size()
        
        with torch.no_grad():

            _, mu_base, _, _ = self.model(
                input_ids, 
                attention_mask, 
                sentence_mask, 
                input_ids.view(B, S*W)
            )
            
            attributions = []
            for i in range(S):
                if sentence_mask[0, i] == 0:
                    continue
                

                occ_sentence_mask = sentence_mask.clone()
                occ_sentence_mask[0, i] = 0 
                

                _, mu_occ, _, _ = self.model(
                    input_ids, 
                    attention_mask, 
                    occ_sentence_mask, 
                    input_ids.view(B, S*W)
                )
                

                importance_score = torch.norm(mu_base - mu_occ, p=2).item()
                attributions.append({'sentence_idx': i, 'importance': importance_score})
                
        return sorted(attributions, key=lambda x: x['importance'], reverse=True)

    def counterfactual_audit(self, original_inputs, counterfactual_inputs, threshold=0.3):
        """
        Compares original resume inputs with gender/demographic-swapped counterfactuals.
        Measures cosine similarity and L2 distance shift in latent space.
        """
        self.model.eval()
        with torch.no_grad():
            orig_ids, orig_attn, orig_smask = original_inputs
            cf_ids, cf_attn, cf_smask = counterfactual_inputs
            
            B_o, S_o, W_o = orig_ids.size()
            B_c, S_c, W_c = cf_ids.size()

            _, mu_orig, _, _ = self.model(
                orig_ids, 
                orig_attn, 
                orig_smask, 
                orig_ids.view(B_o, S_o*W_o)
            )
            _, mu_cf, _, _ = self.model(
                cf_ids, 
                cf_attn, 
                cf_smask, 
                cf_ids.view(B_c, S_c*W_c)
            )
            
            cos_sim = F.cosine_similarity(mu_orig, mu_cf).item()
            l2_dist = torch.norm(mu_orig - mu_cf, p=2).item()
            
            is_biased = l2_dist > threshold
            
            return {
                "cosine_similarity": cos_sim,
                "l2_distance": l2_dist,
                "bias_detected": is_biased
            }
