"""
FCFA Framework - Function-word Cross-lingual Framework for Alignment
A comprehensive class for handling word alignment, visualization, and divergence analysis.
"""

import pickle
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm


class FCFAFramework:
    """
    Main class for handling cross-lingual alignment analysis with function words.
    
    Attributes:
        src_tokens (Dict[int, str]): Source language tokens
        tgt_tokens (Dict[int, str]): Target language tokens
        alignments (List[Tuple[int, int]]): Word alignment pairs
        srl_tags (Optional[List[str]]): Semantic role labels
    """
    
    def __init__(self):
        """Initialize the FCFA Framework."""
        self.src_tokens: Dict[int, str] = {}
        self.tgt_tokens: Dict[int, str] = {}
        self.alignments: List[Tuple[int, int]] = []
        self.srl_tags: Optional[List[str]] = None
        self.function_words: List[str] = []
        
    @staticmethod
    def load_pkl(filepath: str) -> Any:
        """
        Load a pickle file.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Loaded pickle object
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_pkl(obj: Any, filepath: str) -> None:
        """
        Save object to pickle file.
        
        Args:
            obj: Object to save
            filepath: Output file path
        """
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    
    def load_alignment_output(self, align_output: str, idx_token: Tuple[Dict, Dict]) -> None:
        """
        Load alignment output and token mappings.
        
        Args:
            align_output: String of alignment pairs (e.g., "0-0 1-1 2-3")
            idx_token: Tuple of (src_tokens_dict, tgt_tokens_dict)
        """
        self.alignments = [
            (int(item.split('-')[0]), int(item.split('-')[1])) 
            for item in align_output.split()
        ]
        self.src_tokens = idx_token[0]
        self.tgt_tokens = idx_token[1]
    
    def get_src_token_indices(self) -> List[int]:
        """Get list of source token indices from alignments."""
        return [item[0] for item in self.alignments]
    
    def get_tgt_token_indices(self) -> List[int]:
        """Get list of target token indices from alignments."""
        return [item[1] for item in self.alignments]
    
    def find_one_to_many(self) -> List[int]:
        """
        Find source indices that align to multiple target tokens (one-to-many).
        
        Returns:
            List of source indices with multiple alignments
        """
        src_counter = Counter(self.get_src_token_indices())
        return [key for key, value in src_counter.items() if value > 1]
    
    def find_many_to_one(self) -> List[int]:
        """
        Find target indices that have multiple source alignments (many-to-one).
        
        Returns:
            List of target indices with multiple alignments
        """
        tgt_counter = Counter(self.get_tgt_token_indices())
        return [key for key, value in tgt_counter.items() if value > 1]
    
    def get_one_to_many_groups(self, one_to_many_idx: List[int]) -> List[List[Tuple[int, int]]]:
        """
        Group one-to-many alignments.
        
        Args:
            one_to_many_idx: List of source indices with multiple alignments
            
        Returns:
            List of grouped alignment tuples, sorted by target index
        """
        result = []
        for idx in one_to_many_idx:
            temp = [(src, tgt) for (src, tgt) in self.alignments if src == idx]
            temp.sort(key=lambda x: x[1])
            result.append(temp)
        return result
    
    def get_many_to_one_groups(self, many_to_one_idx: List[int]) -> List[List[Tuple[int, int]]]:
        """
        Group many-to-one alignments.
        
        Args:
            many_to_one_idx: List of target indices with multiple alignments
            
        Returns:
            List of grouped alignment tuples, sorted by source index
        """
        result = []
        for idx in many_to_one_idx:
            temp = [(src, tgt) for (src, tgt) in self.alignments if tgt == idx]
            temp.sort(key=lambda x: x[0])
            result.append(temp)
        return result
    
    def visualize_alignment_html(self, output_filepath: str, src_flag: bool = True) -> Tuple[List, List]:
        """
        Generate HTML visualization of word alignments.
        
        Args:
            output_filepath: Path to output HTML file
            src_flag: If True, process src-tgt; if False, process tgt-src
            
        Returns:
            Tuple of (one_to_many_groups, many_to_one_groups)
        """
        flag_val = 0 if src_flag else 1
        
        one_to_many_idx = self.find_one_to_many()
        many_to_one_idx = self.find_many_to_one()
        
        final_otm = self.get_one_to_many_groups(one_to_many_idx)
        final_mto = self.get_many_to_one_groups(many_to_one_idx)
        
        with open(output_filepath, 'w', encoding='utf-8') as fp:
            for src_token_key in self.src_tokens.keys():
                if src_token_key in self.get_src_token_indices():
                    decoded_only = [item for item in self.alignments if item[flag_val] == src_token_key]
                    decoded_only.sort(key=lambda x: x[1])
                    
                    for src, tgt in decoded_only:
                        # Skip if indices are out of bounds
                        if src not in self.src_tokens or tgt not in self.tgt_tokens:
                            continue
                            
                        if src in one_to_many_idx:
                            color = "orange"
                        elif tgt in many_to_one_idx:
                            color = "purple"
                        else:
                            color = "blue"
                        
                        html = f'''<!DOCTYPE html>
<html>
<body>
<p style="color:{color};">[{self.src_tokens[src]}] <span style="color:black;"> &#11835</span>
<span style="color:{color};">[{self.tgt_tokens[tgt]}]</span>
<span style="color:{color};">;</span>
<span style="color:{color};">{src}-{tgt}</span></p>
</body>
</html>'''
                        fp.write(html)
                else:
                    html = f'''<!DOCTYPE html>
<html>
<body>
<p style="color:gray;">[{self.src_tokens[src_token_key]}] <span style="color:black;"> &#11835</span>
<span style="color:gray;">\u03B5</span></p>
</body>
</html>'''
                    fp.write(html)
        
        return final_otm, final_mto
    
    def visualize_with_srl(self, tag_list: List[str], output_filepath: str, 
                          sent_index: int = 0, src_flag: bool = True) -> List:
        """
        Generate HTML visualization with SRL tags.
        
        Args:
            tag_list: List of semantic role labels
            output_filepath: Path to output HTML file
            sent_index: Sentence index for debugging
            src_flag: If True, process src-tgt; if False, process tgt-src
            
        Returns:
            List of saved phrase alignments
        """
        self.srl_tags = tag_list
        flag_val = 0 if src_flag else 1
        
        one_to_many_idx = self.find_one_to_many()
        many_to_one_idx = self.find_many_to_one()
        
        save_phrase = []
        
        with open(output_filepath, 'w', encoding='utf-8') as fp:
            for src_token_key in self.src_tokens.keys():
                if src_token_key in self.get_src_token_indices():
                    decoded_only = [item for item in self.alignments if item[flag_val] == src_token_key]
                    decoded_only.sort(key=lambda x: x[1])
                    save_phrase.append(decoded_only)
                    
                    for src, tgt in decoded_only:
                        srl_label = tag_list[src] if tag_list[src] == 'O' else tag_list[src]
                        
                        if src in one_to_many_idx:
                            color = "orange"
                        elif tgt in many_to_one_idx:
                            color = "purple"
                        else:
                            color = "blue"
                        
                        html = f'''<!DOCTYPE html>
<html>
<body>
<p style="color:{color};">[{srl_label}-{self.src_tokens[src]}] <span style="color:black;"> &#11835</span>
<span style="color:{color};">[{self.tgt_tokens[tgt]}]</span>
<span style="color:orange;">;</span>
<span style="color:orange;">{src}-{tgt}</span></p>
</body>
</html>'''
                        fp.write(html)
        
        return save_phrase
    
    @staticmethod
    def get_tgt_src_candidates(alignments: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """
        Create target-to-source candidate mapping.
        
        Args:
            alignments: List of (src, tgt) alignment pairs
            
        Returns:
            Dictionary mapping target indices to list of source indices
        """
        tgt_src_candi = {}
        for src, tgt in alignments:
            if tgt not in tgt_src_candi:
                tgt_src_candi[tgt] = [src]
            else:
                tgt_src_candi[tgt].append(src)
        return tgt_src_candi
    
    @staticmethod
    def get_src_tgt_candidates(alignments: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """
        Create source-to-target candidate mapping.
        
        Args:
            alignments: List of (src, tgt) alignment pairs
            
        Returns:
            Dictionary mapping source indices to list of target indices
        """
        src_tgt_candi = {}
        for src, tgt in alignments:
            if src not in src_tgt_candi:
                src_tgt_candi[src] = [tgt]
            else:
                src_tgt_candi[src].append(tgt)
        return src_tgt_candi
    
    @staticmethod
    def get_only_tag_list(srl_dict: Dict) -> List[List[str]]:
        """
        Extract only tag lists from SRL dictionary.
        
        Args:
            srl_dict: AllenNLP SRL dictionary with 'verbs' key
            
        Returns:
            List of tag lists, one per verb
        """
        return [item['tags'] for item in srl_dict['verbs']]
    
    @staticmethod
    def get_phraselist(tag_list: List[str]) -> List[List[Tuple]]:
        """
        Convert tag list into phrase list with B-/I-/O grouping.
        
        Args:
            tag_list: List of SRL tags (e.g., ['B-ARG0', 'I-ARG0', 'B-V', ...])
            
        Returns:
            List of phrases, each phrase is list of (tag, idx) tuples
        """
        temp = []
        phrase_list = []
        
        for idx, tag in enumerate(tag_list):
            if idx < len(tag_list) - 1:
                # Only one 'B' (beginning)
                if tag[0] == 'B' and tag_list[idx+1][0] != 'I':
                    temp.append((tag, idx))
                    phrase_list.append(temp)
                    temp = []
                # 'I' (inside) next to 'B'
                elif tag[0] == 'B' and tag_list[idx+1][0] == 'I':
                    temp.append((tag, idx))
                elif tag[0] == 'I' and tag_list[idx+1][0] == 'I':
                    temp.append((tag, idx))
                elif tag[0] == 'I' and tag_list[idx+1][0] != 'I':
                    temp.append((tag, idx))
                    phrase_list.append(temp)
                    temp = []
                elif tag[0] == 'O':
                    temp.append((tag, idx))
                    phrase_list.append(temp)
                    temp = []
            else:
                # Last word
                temp.append((tag, idx))
                phrase_list.append(temp)
                temp = []
        
        return phrase_list
    
    @staticmethod
    def get_divergence_sets(candidates: Dict[int, List[int]]) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        Separate one-to-one alignments from divergent alignments.
        
        Args:
            candidates: Dictionary mapping indices to lists of alignments
            
        Returns:
            Tuple of (one_to_one_dict, divergence_dict)
        """
        onetoone = {}
        divergence = {}
        
        for key, val in candidates.items():
            if len(val) == 1:
                onetoone[key] = val[0]
            else:
                divergence[key] = val
        
        return onetoone, divergence
    
    @staticmethod
    def find_function_words(indices: List[int], token_list: Dict[int, str], 
                           fwords_list: List[str]) -> List[int]:
        """
        Find function words in given indices.
        
        Args:
            indices: List of token indices to check
            token_list: Dictionary mapping indices to tokens
            fwords_list: List of function words
            
        Returns:
            List of indices that are function words
        """
        final_item = []
        for idx in indices:
            if token_list[idx] in fwords_list:
                final_item.append(idx)
        return final_item
    
    def get_alignment_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current alignments.
        
        Returns:
            Dictionary with alignment statistics
        """
        return {
            'total_alignments': len(self.alignments),
            'src_tokens_count': len(self.src_tokens),
            'tgt_tokens_count': len(self.tgt_tokens),
            'one_to_many_count': len(self.find_one_to_many()),
            'many_to_one_count': len(self.find_many_to_one()),
            'one_to_one_count': len(self.alignments) - len(self.find_one_to_many()) - len(self.find_many_to_one())
        }
    
    @staticmethod
    def get_only_tgt(reordered_phrase: Dict) -> List[int]:
        """
        Extract only target indices from reordered phrase dictionary.
        
        Args:
            reordered_phrase: Dictionary of reordered phrases
            
        Returns:
            List of target indices
        """
        only_tgt = []
        for tgts in reordered_phrase.values():
            for tgt in tgts:
                only_tgt.append(tgt[1])
        return only_tgt
    
    @staticmethod
    def get_sort_dict(range_phrase_set: List) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        Sort and categorize phrase alignments into one-to-one and many-to-one.
        
        Args:
            range_phrase_set: Set of phrase ranges
            
        Returns:
            Tuple of (onetoone_dict, manytoone_dict)
        """
        final_ex = []
        for item in range_phrase_set:
            if len(item) > 1:
                for align in item[2]:
                    final_ex.append(align)
            else:
                final_ex.append(item[0])
        
        tgt_src_candi_dict = FCFAFramework.get_tgt_src_candidates(final_ex)
        
        onetoone = {}
        manytoone = {}
        for key, val in tgt_src_candi_dict.items():
            if len(val) == 1:
                onetoone[key] = val[0]
            else:
                manytoone[key] = val
        
        return onetoone, manytoone
    
    @staticmethod
    def get_new_align_fcfa(fwords_list: List[str], token_idx_list: Dict[int, str],
                          src_tgt_candi: Dict, tgt_src_candi: Dict,
                          src_range: List[int], tgt_range: List[int],
                          all_manyone: Dict, only_oneone_tgt: List[int],
                          head_initial: bool = True) -> List[Tuple[int, int]]:
        """
        Get new alignment using FCFA algorithm with function word filtering.
        
        Args:
            fwords_list: List of function words
            token_idx_list: Dictionary mapping indices to tokens
            src_tgt_candi: Source-to-target candidates
            tgt_src_candi: Target-to-source candidates
            src_range: Source range indices
            tgt_range: Target range indices
            all_manyone: All many-to-one mappings
            only_oneone_tgt: Only one-to-one target indices
            head_initial: Whether to use head-initial ordering
            
        Returns:
            List of (src, tgt) alignment tuples
        """
        ts_oneone, manytoone = FCFAFramework.get_divergence_sets(tgt_src_candi)
        st_oneone, onetomany = FCFAFramework.get_divergence_sets(src_tgt_candi)
        
        # Intersection between two oneone (tgt_src and src_tgt)
        ts = list(ts_oneone.values())
        st = list(st_oneone.keys())
        ts_st = [src for src in ts if src in st]
        
        onetoone = {tgt: src[0] for tgt, src in tgt_src_candi.items() if src[0] in ts_st}
        
        new_onetoone = {}
        
        for tgt, src in onetoone.items():
            if (src in src_range) and (tgt not in all_manyone.keys()):
                new_onetoone[tgt] = src
        
        # If no divergence
        if len(manytoone) == 0 and len(onetomany) == 0:
            sorted_onetoone = sorted(new_onetoone.items(), key=lambda x: x[1], reverse=False)
            updated_onetoone = [(items[1], items[0]) for items in sorted_onetoone]
            return updated_onetoone
        
        # Handle one-to-many
        for a_src, tgts in onetomany.items():
            onemany_val = deepcopy(tgts)
            
            for tgt in onemany_val[:]:
                if tgt not in tgt_range:
                    onemany_val.remove(tgt)
                elif tgt in only_oneone_tgt:
                    onemany_val.remove(tgt)
            
            if len(onemany_val) == 1:
                new_onetoone[onemany_val[0]] = a_src
            else:
                for tgt in onemany_val:
                    new_onetoone[tgt] = a_src
        
        # Handle many-to-one
        for key, val in manytoone.items():
            temp_val = deepcopy(val)
            
            # Check range
            for val_item in temp_val[:]:
                if val_item not in src_range:
                    temp_val.remove(val_item)
            
            # Check one-to-one
            for val_item in temp_val[:]:
                if val_item in new_onetoone.values():
                    temp_val.remove(val_item)
            
            if len(temp_val) == 1:
                new_onetoone[key] = temp_val[0]
            else:
                # Remove function words
                fwords_idx = FCFAFramework.find_function_words(temp_val, token_idx_list, fwords_list)
                
                if len(fwords_idx) != len(temp_val) and len(fwords_idx) > 0:
                    for f_idx in fwords_idx:
                        if f_idx in temp_val:
                            temp_val.remove(f_idx)
                
                if len(temp_val) == 1:
                    new_onetoone[key] = temp_val[0]
                elif len(temp_val) > 1:
                    # Head initial/final
                    if head_initial:
                        new_onetoone[key] = temp_val[0]
                    else:
                        new_onetoone[key] = temp_val[:2][1] if len(temp_val) >= 2 else temp_val[0]
                # If temp_val is empty, skip this alignment
        
        sorted_onetoone = sorted(new_onetoone.items(), key=lambda x: x[1], reverse=False)
        updated_onetoone = [(items[1], items[0]) for items in sorted_onetoone]
        return updated_onetoone
    
    @staticmethod
    def get_all_align_dict(rephrase: List, src_tgt_idx: Tuple[Dict, Dict]) -> List[Tuple]:
        """
        Get all alignments from rephrase results.
        
        Args:
            rephrase: Rephrase results (list of tuples)
            src_tgt_idx: Tuple of (src_tokens, tgt_tokens)
            
        Returns:
            List of (src, tgt) alignment tuples
        """
        final_phrase_only = []
        
        for phrase_tuple in rephrase:
            # phrase_tuple is (src_first, tgt_first)
            if len(phrase_tuple) >= 2:
                src_first = phrase_tuple[0]
                # Get alignments from src_first
                for item in src_first:
                    if isinstance(item, tuple) and len(item) >= 2:
                        final_phrase_only.append((item[0], item[1]))
            elif len(phrase_tuple) == 1:
                # Single alignment
                if isinstance(phrase_tuple[0], list) and len(phrase_tuple[0]) > 0:
                    item = phrase_tuple[0][0]
                    if isinstance(item, tuple) and len(item) >= 2:
                        final_phrase_only.append((item[0], item[1]))
        
        # Add epsilon for unaligned source tokens
        for idx in range(len(src_tgt_idx[0])):
            only_src_from_final_phrase = [pair[0] for pair in final_phrase_only]
            if idx not in only_src_from_final_phrase:
                final_phrase_only.append((idx, 'eps'))
        
        final_phrase_only.sort(key=lambda x: x[0])
        return final_phrase_only
    
    @staticmethod
    def get_fcfa_align_eval(src_tgt_idx: Tuple[Dict, Dict], final_rephrase_only: List,
                           srl_dict_verbs: Dict) -> Tuple[List[str], List[Tuple]]:
        """
        Get FCFA alignment for evaluation.
        
        Args:
            src_tgt_idx: Tuple of (src_tokens, tgt_tokens)
            final_rephrase_only: Final rephrase alignments
            srl_dict_verbs: SRL verb dictionary
            
        Returns:
            Tuple of (src_label_list, tgt_label_list)
        """
        src_idx_tokens = src_tgt_idx[0]
        tgt_idx_tokens = src_tgt_idx[1]
        tag_list = srl_dict_verbs['tags']
        
        src_label_list = []
        tgt_label_list = []
        
        for align in final_rephrase_only:
            src_idx = align[0]
            tgt_idx = align[1]
            
            # Bounds checking for source
            if src_idx not in src_idx_tokens:
                continue
            
            align_src_token = src_idx_tokens[src_idx]
            align_src_label = tag_list[src_idx] if src_idx < len(tag_list) else 'O'
            
            if align_src_label != 'O':
                align_src_label = align_src_label
            
            if tgt_idx == 'eps':
                align_tgt_token = '\u03B5'
                align_tgt_label = 'n'
            else:
                # Bounds checking for target
                if tgt_idx not in tgt_idx_tokens:
                    continue
                align_tgt_token = tgt_idx_tokens[tgt_idx]
                align_tgt_label = align_src_label
            
            src_label_list.append(align_src_label)
            tgt_label_list.append((tgt_idx, align_tgt_label))
        
        return src_label_list, tgt_label_list
    
    @staticmethod
    def get_phrase_object(eval_data: Dict, idx_token: List) -> List[Dict]:
        """
        Create phrase object for evaluation.
        
        Args:
            eval_data: Evaluation data dictionary
            idx_token: Index token mappings
            
        Returns:
            List of phrase objects
        """
        new_object = []
        
        for idx, (sents, labels) in enumerate(eval_data.items()):
            tgt_idx_token = idx_token[idx]
            tgt_idxs = list(tgt_idx_token[1].keys())
            tgt_tokens = list(tgt_idx_token[1].values())
            
            new_temp_dict = {'verbs': []}
            
            for label_list in labels:
                temp_verb_dict = {}
                idx_label_dict = {idx: label for idx, label in label_list[1]}
                
                if 'B-V' in idx_label_dict.values():
                    verb_idx = [idx for idx, label in idx_label_dict.items() if label == 'B-V'][0]
                    verb = tgt_idx_token[1][verb_idx]
                else:
                    verb = '_'
                
                new_tgt_all_label = [
                    idx_label_dict[idx] if idx in idx_label_dict else 'O' 
                    for idx in tgt_idxs
                ]
                
                temp_verb_dict['verb'] = verb
                description = [
                    f"[{label}-{token}]" if label != 'O' else token
                    for token, label in zip(tgt_tokens, new_tgt_all_label)
                ]
                temp_verb_dict['description'] = ' '.join(description)
                temp_verb_dict['tags'] = new_tgt_all_label
                
                new_temp_dict['verbs'].append(temp_verb_dict)
            
            new_temp_dict['words'] = list(tgt_tokens)
            new_temp_dict['sent'] = ' '.join(tgt_tokens)
            new_object.append(new_temp_dict)
        
        return new_object


def batch_process_alignments(align_outputs: List[str], 
                             idx_tokens_list: List[Tuple[Dict, Dict]]) -> List[FCFAFramework]:
    """
    Process multiple alignments in batch.
    
    Args:
        align_outputs: List of alignment output strings
        idx_tokens_list: List of token dictionaries
        
    Returns:
        List of FCFAFramework instances
    """
    frameworks = []
    for align_out, idx_tokens in zip(align_outputs, idx_tokens_list):
        framework = FCFAFramework()
        framework.load_alignment_output(align_out, idx_tokens)
        frameworks.append(framework)
    return frameworks


if __name__ == '__main__':
    # Example usage with your file paths
    import os
    
    # File paths from your notebook
    sents_only_fp = "/blue/bonniejdorr/youms/SRL/data/conll_data/text_only/conll_file/allen_phrasal_fcfa/en_fr_src_tgt.src-tgt"
    out_path = '/blue/bonniejdorr/youms/SRL/data/conll_data/text_only/conll_file/allen_phrasal_fcfa/en_fr_src_tgt.out'
    idx_token_path = '/blue/bonniejdorr/youms/SRL/data/conll_data/text_only/conll_file/allen_phrasal_fcfa/en_fr_idx_tokens_2046.pkl'
    allen_srl_path = '/blue/bonniejdorr/youms/SRL/data/conll_data/text_only/conll_file/allen_phrasal_results/EN_objs_xsrl_phrasal.pkl'
    fwords_path = '/blue/bonniejdorr/youms/SRL/datasets_final/en_function_words.pkl'
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [sents_only_fp, out_path, idx_token_path, allen_srl_path]):
        print("Using example data (files not found)")
        
        # Example data
        framework = FCFAFramework()
        
        src_tokens = {0: "Hello", 1: "world", 2: "."}
        tgt_tokens = {0: "Bonjour", 1: "le", 2: "monde", 3: "."}
        idx_token = (src_tokens, tgt_tokens)
        align_output = "0-0 1-1 1-2 2-3"
        
        framework.load_alignment_output(align_output, idx_token)
        
        stats = framework.get_alignment_statistics()
        print("Alignment Statistics:", stats)
        
        print("One-to-many indices:", framework.find_one_to_many())
        print("Many-to-one indices:", framework.find_many_to_one())
    else:
        print("Loading real data...")
        
        # Load data
        with open(sents_only_fp, 'r', encoding='utf-8') as f:
            sents_only = [line.strip() for line in f.readlines()]
        
        with open(out_path, 'r', encoding='utf-8') as f:
            align_outputs = [line.strip() for line in f.readlines()]
        
        idx_tokens = FCFAFramework.load_pkl(idx_token_path)
        allen_srl_dict = FCFAFramework.load_pkl(allen_srl_path)
        fwords_list = FCFAFramework.load_pkl(fwords_path) if os.path.exists(fwords_path) else []
        
        print(f"Loaded {len(sents_only)} sentences")
        print(f"Loaded {len(align_outputs)} alignments")
        print(f"Loaded {len(idx_tokens)} token pairs")
        print(f"Loaded {len(allen_srl_dict)} SRL entries")
        
        # Process first example
        if len(align_outputs) > 0:
            framework = FCFAFramework()
            framework.load_alignment_output(align_outputs[0], idx_tokens[0])
            framework.function_words = fwords_list
            
            print("\nFirst sentence:", sents_only[0])
            print("Statistics:", framework.get_alignment_statistics())
            print("One-to-many:", framework.find_one_to_many())
            print("Many-to-one:", framework.find_many_to_one())

