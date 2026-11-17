import openmc
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import argparse
from branching_adder import MT_TO_REACTION
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--branching-energy", type=float, required=True, help="Incident neutron energy [eV] at which to evaluate isomer branching yields")

BRANCHING_E = parser.parse_args().branching_energy

FOLDER_PATH = pathlib.Path("/opt/marathon/openmc/nucleardata")
NEUTRON_FILES = pathlib.Path.glob(FOLDER_PATH, "*.jeff")
CHAIN_PATH = os.path.join(FOLDER_PATH, "chain_jeff40_fast.xml")
SYMBOLS = openmc.data.ATOMIC_SYMBOL

# All MT numbers from OpenMC REACTIONS dictionary
list_of_mt_to_extract = [
    # Individual MT numbers
    11, 16, 17, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 41, 42, 44, 45,
    102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
]

def expected_products_no_branching(mt: int) -> int:
    """Given the assumption of no isomeric products,
    how many unique products are expected from a given reaction"""
    mt_products = {11: 2,
                    16: 1,
                    17: 1,
                    22: 2,
                    23: 2,
                    24: 2,
                    25: 2,
                    28: 2,
                    29: 2,
                    30: 2,
                    32: 2,
                    33: 2,
                    34: 2,
                    35: 3,
                    36: 3,
                    37: 1,
                    41: 2,
                    42: 2,
                    44: 2,
                    45: 3,
                    102: 0,
                    103: 1,
                    104: 1,
                    105: 1,
                    106: 1,
                    107: 1,
                    108: 1,
                    109: 1,
                    111: 1,
                    112: 2,
                    113: 2,
                    114: 2,
                    115: 2,
                    116: 2,
                    117: 2,
                    152: 1,
                    153: 1,
                    154: 2,
                    155: 2,
                    160: 1,
                    161: 1,
                    162: 2,
                    163: 2,
                    164: 2,
                    165: 2,
                    166: 2,
                    167: 2,
                    168: 2,
                    169: 2,
                    170: 2,
                    171: 2,
                    172: 2,
                    173: 2,
                    174: 2,
                    175: 2,
                    176: 2,
                    177: 2,
                    178: 2,
                    180: 2,
                    181: 3,
                    182: 2,
                    183: 3,
                    184: 3,
                    185: 3,
                    186: 3,
                    187: 3,
                    188: 3,
                    189: 3,
                    190: 2,
                    191: 2,
                    192: 2,
                    193: 2,
                    194: 2,
                    195: 2,
                    196: 3,
                    197: 1,
                    198: 2,
                    }
    
    return mt_products[mt] + 2 #product nucleus and photon not included in initial numbers    

def extract_reactions(chain_path):
    root = ET.parse(chain_path).getroot()
    out = []
    for nuc in root.iter("nuclide"):
        z, a, m = openmc.data.zam(nuc.get("name"))          # "Am242_m1" -> (95, 242, 1)
        rxns = tuple((r.get("type").strip("()"), r.get("target"), float(r.get("branching_ratio", 1.0))) for r in nuc.findall("reaction"))
        if rxns:
            out.append([(z, a, m), rxns])
    return out

def get_neutron_file_name(z,a,m):
    if m == 0:
        mtag = "g"
    elif m == 1:
        mtag = "m"
    elif m == 2:
        mtag = "n"
    else:
        raise ValueError(f"Unexpectedly high metastable state: {(z,a,m)}")
    neutron_file_name = f"n_{z}-{SYMBOLS[z]}-{a:03}{mtag}.jeff"
    return neutron_file_name


reactions = extract_reactions(CHAIN_PATH)

nuclides_from_chain = [zam for zam, rxns in reactions]
print(f"{len(nuclides_from_chain)} nuclide-isomer pairs with instant neutron reactions located from full chain file.")

nuclides_with_reactions = []

for nuclide in nuclides_from_chain:
    (z,a,m) = nuclide
    file_path = FOLDER_PATH.joinpath(get_neutron_file_name(z,a,m))
    if file_path.exists(): 
        nuclides_with_reactions.append(nuclide)

print(f"{len(nuclides_with_reactions)} nuclide-isomer pairs cross-referenced with incident neutron data. {len(nuclides_from_chain)-len(nuclides_with_reactions)} nuclide-isomer pairs removed.")

files = []
for nuclide in nuclides_with_reactions:
    (z,a,m) = nuclide
    file = get_neutron_file_name(z,a,m)
    file_path = FOLDER_PATH.joinpath(file)
    files.append(file_path)

evals = []
for file in files:
    print(f"File {file} processing into material evaluation data...")
    evals.append(openmc.data.endf.get_evaluations(file))
    
to_be_changed = []
for ev in evals:
    incident_neutron_data = openmc.data.IncidentNeutron.from_endf(ev[0])
    reactions = incident_neutron_data.reactions
    print(f"Extracting reactions for {incident_neutron_data.name}...")
    changed_reactions = defaultdict(list)
    for mt in list_of_mt_to_extract:
        if mt in reactions:
            reaction = reactions[mt]
            #if mt != 102 and len(reaction.products) != expected_products_no_branching(mt):
            if len(reaction.products) != expected_products_no_branching(mt):
                #print(reaction, reaction.products)
                isomers = [product for product in reaction.products if '_e' in product.particle]# and product.yield_(BRANCHING_E) > 1e-3]
                ground_state_yield = 1
                for isomer in isomers:
                    isomer_yield = float(isomer.yield_(BRANCHING_E))
                    ground_state_yield -= isomer_yield
                    #print(isomer, isomer_yield)
                    changed_reactions[mt].append([isomer.particle, isomer_yield])
                changed_reactions[mt].append(['ground state', ground_state_yield])

    z = incident_neutron_data.atomic_number
    a = incident_neutron_data.mass_number
    m = incident_neutron_data.metastable

    to_be_changed.append((z,a,m,changed_reactions))

# ============================================================================
# Write isomer branching into a copy of the chain file
# ============================================================================
import re
import shutil

MODDED_CHAIN_PATH = os.path.join(FOLDER_PATH, "chain_jeff40_fast_modded.xml")
shutil.copyfile(CHAIN_PATH, MODDED_CHAIN_PATH)  # overwrites any existing modded file
print(f"Copied {CHAIN_PATH} -> {MODDED_CHAIN_PATH}")

tree = ET.parse(MODDED_CHAIN_PATH)
chain_root = tree.getroot()
chain_nuclides = {nuc.attrib["name"]: nuc for nuc in chain_root.iter("nuclide")}

stats = defaultdict(int)
skipped = []  # (parent, reaction, reason)

for (z, a, m, changed) in to_be_changed:
    parent = openmc.data.gnds_name(z, a, m)
    nuc = chain_nuclides.get(parent)
    if nuc is None:
        stats["parent not in chain"] += 1
        continue

    for mt, branches in changed.items():
        rtype = MT_TO_REACTION.get(f"MT_{mt}")
        if rtype is None:
            stats["MT not in MT_TO_REACTION"] += 1
            continue

        existing = [r for r in nuc.findall("reaction") if r.get("type") == rtype]
        if not existing:
            stats["reaction not in chain"] += 1   # only replace reactions already present
            continue

        isomer_branches = [(p, float(y)) for p, y in branches if p != "ground state"]
        if not isomer_branches:
            stats["ground-state only, left unchanged"] += 1
            continue
        ground_yield = float(next(y for p, y in branches if p == "ground state"))

        # 'Ir192_e3' -> product 'Ir192', level 3
        product = isomer_branches[0][0].rsplit("_e", 1)[0]
        ground_line = next((r for r in existing if r.get("target") == product), None)
        if ground_line is None:
            skipped.append((parent, rtype, f"no existing line targets {product}"))
            continue

        # Chain isomers sorted by state number: [Ir192_m1, Ir192_m2, ...]
        chain_isomers = sorted((n for n in chain_nuclides if re.fullmatch(rf"{product}_m\d+", n)),
                               key=lambda n: int(n.rsplit("_m", 1)[1]))
        if len(chain_isomers) == 0:
            # No isomer in chain -> fold isomer yield into ground state (= 100% ground, so chain is unchanged)
            stats["isomer yields folded into ground state"] += len(isomer_branches)
            continue

        # Sum yields per ENDF level, then rank levels: lowest level -> _m1, next -> _m2, ...
        level_yields = defaultdict(float)
        for p, y in isomer_branches:
            level_yields[int(p.rsplit("_e", 1)[1])] += y
        levels = sorted(level_yields)
        if len(levels) != len(chain_isomers):
            skipped.append((parent, rtype, f"{len(levels)} ENDF levels {levels} vs {len(chain_isomers)} chain isomers {chain_isomers}"))
            continue
        isomer_lines = [(iso, level_yields[lvl]) for lvl, iso in zip(levels, chain_isomers)]

        # Replace all existing lines of this reaction type with ground + isomer lines
        insert_at = list(nuc).index(existing[0])
        q_value = ground_line.attrib["Q"]
        for r in existing:
            nuc.remove(r)

        new_lines = [(product, ground_yield)] + isomer_lines
        for i, (target, br) in enumerate(new_lines):
            el = ET.Element("reaction", {"type": rtype, "Q": q_value, "target": target,
                                         "branching_ratio": repr(br)})
            nuc.insert(insert_at + i, el)

        nuc.set("reactions", str(len(nuc.findall("reaction"))))
        stats["reactions replaced"] += 1
        #print(f"  {parent} {rtype}: " + ", ".join(f"{t} {br:.4f}" for t, br in new_lines))

ET.indent(tree, space="  ")  # re-indent so inserted lines match the 2-space layout of the original
tree.write(MODDED_CHAIN_PATH, encoding="utf-8")
with open(MODDED_CHAIN_PATH, "a") as f:
    f.write("\n")

print(f"\nWrote {MODDED_CHAIN_PATH}")
for key, count in stats.items():
    print(f"  {key}: {count}")
print(f"{stats['isomer yields folded into ground state']} isomer yields combined with ground state yields due to missing isomer in chain")
if skipped:
    print(f"  skipped: {len(skipped)}")
    for parent, rtype, reason in skipped:
        print(f"    {parent} {rtype}: {reason}")

