#!/usr/bin/env python3
"""
Download and prepare necessary data files for the receptor design pipeline.
"""

import os
import json
from pathlib import Path


# Ensure required directories exist
def create_dirs():
    dirs = [
        "data/sequences",
        "data/expert_knowledge"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


# Create placeholder sequence files
def create_sequence_files():
    # GPCR Class A
    gpcr_content = """>P35348|ADRA1A_HUMAN Alpha-1A adrenergic receptor
MVFLSGNASDSSNCTQPPAPVNISKAILLGVILGGLILFGVLGNILVILSVACHRHLHS
VTHYYIVNLAVADLLLTSTVLPFSAIFEVLGYWAFGRVFCNIWAAVDVLCCTASILSLC
TISVDRYVGVRHSLYPAIMTERKAAAILALLWAVALVVSVGPLMGWKEIPVDAEEVVGI
EVSQVGPVGASFLTAVWTLCVIAVDRQVVVCFVWRNSRGPSQQSSMFCRCCFVQVVIAV
MLCVGIMVACVMALVHVQLTLRRGWRQPAPGAAAEGCPRRASKPAGPSHAQRELLKTVL
VAAVLLAFILCWLPFSINCVVLFYLKSSRIQADDLTVLALVLSLGLLNAPFDPFCFVSN
VAYLTLGPFQHKLVPPTFNSMINPVIYALRNRDFRYTFQKLVHCCAVATSCTLNPFIYG
FLGKKFKRYLSV
>P35368|ADA1B_HUMAN Alpha-1B adrenergic receptor
MNPDLDTGHNTSAPAHWGELKDDNFTGPNGTAYPGQGSVGAAGGSGGGAAGGGSGGGAI
ASLVIVAALAIVAGNILVILSVACNRHLQTVTNYFIVNLAIADFLGVLAIPFSAIYEVL
GWVWFGRVFCDIWAAFDVMCSTASILNLCVISVDRYVGVRHSLYPAIMCERKAMAILSV
WLISLVISIGPLFGWRQPAAVAEVTVICWSLSAAPPFFNEVSEELACDLHMWTLCVVFF
TIPVVIAVFCYMRVAKVRESHAQRSHVRRVGVAGAGEAEGAARAARGVRAKRVPPASRG
AARTLLLPWVPTTVVTGAVLCLCWAPFVARQAVVETLIGVHHVPSTYNSAFKAFLWLGY
INSGLNPFIYAFFHGEFRKAFQRILCRGDRKRIV"""

    # Cytokine receptors
    cytokine_content = """>P16871|IL7R_HUMAN Interleukin-7 receptor subunit alpha
MGSWSPSMDLLLLLLLGVGVAPGNPLGDSEEPATVPSLGCYWCHVAVFGGLICLLCILF
WVLAQVLLRCSPKVHLIQKGCHDDVAVLSRDCFPMPRYVQRFRADCEVNELRYHPTAWD
GPRSGGDILDYVEVHFPPYLAPARSPTGGREARWEAPGRSRDRAGSALLCSQAAALTAL
LALVAPGTGSFSCYVIEIRSSAIILVPTLWASNPSHRATAHGPREARVHPRAPTEGRWP
PSTPAQEETGCAGDWTEAAHPSCSYQRYRHRSHRALVALRRALWALCRPGSPAETVEIP
APKSLQGIISCLRRDALTLEEKELALLERNKLNCSLDAVLSGLGQYPWSCWVLRGHIPL
RLYRRLRKHFRKNMAYTRDFKPLPEDPRLDMESWVCLALAVLLPLLLGGASGPLCWCLL
LGSRGRAIHWKSGGKCRPGPEEPAGRAAASSWVALHLAPGPGPGCSPRWL"""

    # Ion channels
    ion_content = """>P35498|SCN1A_HUMAN Sodium channel protein type 1 subunit alpha
MAARLLAPPGPDSFKPFTPESLANSVLICFWEALCAIPVCVSGILLWAEQGGPEAAGGD
SGPQERAPGREGVSQAQGLSGPRGPQGCPGSGAPGPGSGSRPLPGSPGHSRPSPALGRG
SFCAFLTLFLFSSLLGAAWGGLGVVRLGALGPGPSSAAESWASAQPGEGLAATRVPPAL
QGATPGTEEPVVAVSWRELFNVVYTAMAEEPFVLKFRSLSDFVTLQSLLCCLVFAFTVV
LMEFQNYVRVLRHPLAWSTVNRGYLSLLQVATFKGWMDIMYAAVDSRNVELQPKMEDLV
DVVSPLICGILVDDHAYWFVEDIIFNSLGIATLALAAQMFRAFIKPLRYWHVVKYIVCT
LYIFTALTILNCVLMAMPTTPTNVEAEGKPAPNSTVTQIATHPYGFTYIALLNITSLIF
TLGFFPLPILRSAWVWDVVIAAVNAVVITLALVGFPGMLTHLLGYRILGGLNLVMGGLT
LIPALGTELTLLITFRIVDRLGIPRPLQFHLVAMLIMDEQGEHPADYWRIVSIFFIIFG
SFFTLNLFIGVIIDNFNQQKKKLGGQDIFMTEEQKKYYNAMKKLGSKKPQKPIPRPLNK"""

    # Write files
    files = {
        "data/sequences/gpcr_class_a.fasta": gpcr_content,
        "data/sequences/cytokine_receptors.fasta": cytokine_content,
        "data/sequences/ion_channels.fasta": ion_content
    }

    for path, content in files.items():
        with open(path, 'w') as f:
            f.write(content)
        print(f"Created sequence file: {path}")


# Create expert knowledge file
def create_expert_knowledge():
    causal_priors = {
        "metadata": {
            "description": "Expert knowledge priors for receptor domain causal relationships",
            "version": "1.0",
            "date": "2025-03-04"
        },
        "nodes": [
            {
                "id": "binding_pocket",
                "type": "structural",
                "description": "The binding pocket structure of the receptor"
            },
            {
                "id": "transmembrane_domain",
                "type": "structural",
                "description": "Transmembrane domain of the receptor"
            },
            {
                "id": "n_terminus",
                "type": "structural",
                "description": "N-terminus of the receptor"
            },
            {
                "id": "c_terminus",
                "type": "structural",
                "description": "C-terminus of the receptor"
            },
            {
                "id": "binding_affinity",
                "type": "functional",
                "description": "Binding affinity to target ligand"
            },
            {
                "id": "specificity",
                "type": "functional",
                "description": "Specificity of binding"
            },
            {
                "id": "stability",
                "type": "biophysical",
                "description": "Thermal and conformational stability"
            },
            {
                "id": "expression",
                "type": "biophysical",
                "description": "Expression level in host organism"
            },
            {
                "id": "signaling",
                "type": "functional",
                "description": "Signaling capability after binding"
            }
        ],
        "edges": [
            {
                "source": "binding_pocket",
                "target": "binding_affinity",
                "confidence": 0.9,
                "mechanism": "direct"
            },
            {
                "source": "binding_pocket",
                "target": "specificity",
                "confidence": 0.85,
                "mechanism": "direct"
            },
            {
                "source": "transmembrane_domain",
                "target": "stability",
                "confidence": 0.8,
                "mechanism": "direct"
            },
            {
                "source": "n_terminus",
                "target": "expression",
                "confidence": 0.7,
                "mechanism": "direct"
            },
            {
                "source": "c_terminus",
                "target": "signaling",
                "confidence": 0.75,
                "mechanism": "direct"
            },
            {
                "source": "stability",
                "target": "expression",
                "confidence": 0.65,
                "mechanism": "indirect"
            },
            {
                "source": "binding_affinity",
                "target": "signaling",
                "confidence": 0.6,
                "mechanism": "indirect"
            },
            {
                "source": "transmembrane_domain",
                "target": "signaling",
                "confidence": 0.55,
                "mechanism": "indirect"
            }
        ]
    }

    # Write file
    path = "data/expert_knowledge/receptor_domain_causal_priors.json"
    with open(path, 'w') as f:
        json.dump(causal_priors, f, indent=2)
    print(f"Created expert knowledge file: {path}")


def main():
    print("Starting data download and preparation...")

    # Create directories
    create_dirs()

    # Create sequence files
    create_sequence_files()

    # Create expert knowledge file
    create_expert_knowledge()

    print("Data preparation completed successfully!")


if __name__ == "__main__":
    main()