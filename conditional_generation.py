import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

from rdkit import RDLogger
from rdkit import Chem

import json
import os

RDLogger.DisableLog('rdApp.*')

from group_selfies import GroupGrammar, Group

device = ("cuda" if torch.cuda.is_available() else "cpu")

def get_property_token_prefix(vals_list, stats_dict, prop_token_dict):
    propval_tokens = prop_token_dict["propval_tokens"]

    property_tokens = []

    for prop, val in vals_list:
        m, std, pmin, pmax = stats_dict[prop]
        prop_val = (val - m) / std

        propval_token_index = int(((prop_val - pmin) / (pmax - pmin)) * len(propval_tokens))

        if propval_token_index < 0:
            propval_token_index = 0
        if propval_token_index >= len(propval_tokens):
            propval_token_index = len(propval_tokens) - 1

        propval_token = propval_tokens[propval_token_index]

        property_tokens.append(prop_token_dict["prop_tokens"][prop])
        property_tokens.append(propval_token)
      
    return ''.join(property_tokens)

def generate_conditional(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    prop_tokens: str,
    n_generate: int
):
    
    model.eval()

    generated_molecules = []
    with torch.no_grad():
        tokenized_props = tokenizer(prop_tokens, add_special_tokens=False)
        input_ids = [tokenizer.bos_token_id] + tokenized_props["input_ids"] # add BOS token
        attention_mask = [1] + tokenized_props["attention_mask"]

        input_ids = torch.tensor(input_ids).to(torch.long)
        attention_mask = torch.tensor(attention_mask).to(torch.long)

        output_ids = model.generate(
            input_ids=input_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device),
            max_length=512,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=n_generate
        )

        for outputs in output_ids:
            gen_mol = tokenizer.decode(outputs, skip_special_tokens=True)
            generated_molecules.append("".join(gen_mol.split()))

    return generated_molecules

def generate_molecules_from_properties(
    model_path: str,
    tpsa: float=None,
    qed: float=None,
    sas: float=None,
    logp: float=None,
    n: int=1000,
    property_tokens_path: str="property_tokens.json",
    vocab_path: str="vocab.json",
    stats_path: str="zinc250k_stats.json"
):

    assert not (tpsa is None and qed is None and sas is None and logp is None), "must add at least one property"

    with open(vocab_path, "r") as f: vocab_dict = json.load(f)["vocab"]

    with open(property_tokens_path, "r") as f: prop_tokens_dict = json.load(f)
    with open(stats_path, "r") as f: stats_dict = json.load(f)

    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(model_path, "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>"
    )

    prop_vals = []
    if tpsa is not None:
        prop_vals.append(("tpsa", tpsa))
    if qed is not None:
        prop_vals.append(("qed", qed))
    if sas is not None:
        prop_vals.append(("sas", sas))
    if logp is not None:
        prop_vals.append(("logp", logp))

    prefix = get_property_token_prefix(
        prop_vals,
        stats_dict,
        prop_tokens_dict
    )

    gs_gen = generate_conditional(
        model.to(device),
        tokenizer,
        prefix,
        n
    )

    grammar = GroupGrammar([Group(fragname, smiles) for fragname, smiles in vocab_dict.items()])

    mols = [grammar.decoder(gs) for gs in gs_gen]
    smiles = [Chem.MolToSmiles(mol) for mol in mols]

    return smiles