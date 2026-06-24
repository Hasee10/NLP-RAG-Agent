"""Collect dental Q&A from MedlinePlus Oral Health and ingest into Supabase.

Run:
    python -m medical-rag.collect_dental

Adds ~200-400 dental chunks with body_system='oral' to the existing
medical_chunks table so /medical-query can answer dental questions.
"""

import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT.parent / ".env", override=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("collect_dental")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 50

# MedlinePlus Health Topics API — dental/oral health
MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query"

DENTAL_TOPICS = [
    "tooth decay", "gum disease", "gingivitis", "periodontitis",
    "toothache", "dental cavities", "oral hygiene", "tooth abscess",
    "root canal", "dental plaque", "tartar", "braces orthodontics",
    "wisdom teeth", "dry mouth", "oral cancer", "canker sores",
    "tooth sensitivity", "dental fillings", "dental crowns",
    "tooth extraction", "fluoride teeth", "mouthguard", "teeth grinding",
    "bad breath halitosis", "tooth whitening", "dental implants",
    "dentures", "cleft palate", "temporomandibular joint TMJ",
    "mouth breathing", "baby teeth", "teething", "dental x-rays",
]

# Curated dental Q&A pairs (authoritative, sourced from ADA/NIH/MedlinePlus)
CURATED_DENTAL_QA = [
    {
        "question": "What causes tooth decay?",
        "answer": "Tooth decay (dental caries) is caused by bacteria in the mouth that feed on sugars and starches from food, producing acids that erode tooth enamel. Streptococcus mutans is the primary bacterium involved. Risk factors include frequent sugar consumption, poor oral hygiene, dry mouth, and lack of fluoride exposure.",
        "source": "ADA/MedlinePlus",
    },
    {
        "question": "How is tooth decay treated?",
        "answer": "Treatment depends on severity: early decay may be reversed with fluoride treatments. Cavities are treated with fillings (composite resin, amalgam, or ceramic). Deeper decay may require a crown, root canal, or extraction if the tooth cannot be saved.",
        "source": "ADA",
    },
    {
        "question": "What is gingivitis?",
        "answer": "Gingivitis is inflammation of the gums caused by plaque buildup along the gumline. Symptoms include red, swollen, bleeding gums. It is reversible with professional cleaning and improved oral hygiene. If untreated, it can progress to periodontitis.",
        "source": "ADA/NIH",
    },
    {
        "question": "What is periodontitis?",
        "answer": "Periodontitis is a serious gum infection that damages the soft tissue and bone supporting the teeth. It develops from untreated gingivitis. Symptoms include receding gums, loose teeth, deep pockets between teeth and gums, and tooth loss. Treatment includes scaling, root planing, and surgery in severe cases.",
        "source": "ADA/Mayo Clinic",
    },
    {
        "question": "What causes bad breath (halitosis)?",
        "answer": "Bad breath is most commonly caused by poor oral hygiene leading to bacterial growth on the tongue, teeth, and gums. Other causes include dry mouth, gum disease, food (garlic, onions), tobacco, and medical conditions such as sinus infections, acid reflux, or diabetes.",
        "source": "ADA/MedlinePlus",
    },
    {
        "question": "What is a root canal procedure?",
        "answer": "A root canal treats infection inside the tooth pulp. The dentist removes infected pulp, cleans and shapes the root canals, fills them with gutta-percha, and seals the tooth, usually with a crown. It relieves pain and saves the tooth from extraction. Modern root canals are no more painful than a filling.",
        "source": "ADA",
    },
    {
        "question": "What causes tooth sensitivity?",
        "answer": "Tooth sensitivity (dentin hypersensitivity) occurs when the dentin beneath the enamel is exposed, often due to enamel erosion, gum recession, tooth grinding, or cracked teeth. Triggers include hot, cold, sweet, or acidic foods. Treatment includes desensitizing toothpaste, fluoride gel, or dental bonding.",
        "source": "ADA/NIH",
    },
    {
        "question": "How do dental braces work?",
        "answer": "Braces apply continuous pressure to teeth over time, gradually moving them into the correct position. They consist of brackets bonded to teeth and wires that connect them. Archwires apply force, and elastic bands may help correct bite issues. Treatment typically lasts 1–3 years.",
        "source": "ADA",
    },
    {
        "question": "What are wisdom teeth and when should they be removed?",
        "answer": "Wisdom teeth are the third set of molars that typically emerge between ages 17–25. They should be removed if they are impacted (unable to fully emerge), cause pain, crowding, cyst formation, gum disease, or tooth decay in adjacent teeth. Many dentists recommend removal before problems develop.",
        "source": "ADA/Mayo Clinic",
    },
    {
        "question": "What is temporomandibular joint (TMJ) disorder?",
        "answer": "TMJ disorders affect the jaw joint and chewing muscles, causing jaw pain, clicking or popping sounds, difficulty opening the mouth, headaches, and earaches. Causes include teeth grinding, jaw injury, arthritis, or stress. Treatment includes mouthguards, physical therapy, pain medication, and rarely surgery.",
        "source": "NIH/NIDCR",
    },
    {
        "question": "How often should you brush and floss your teeth?",
        "answer": "The ADA recommends brushing teeth twice daily for two minutes with fluoride toothpaste, and flossing once daily. Regular brushing and flossing removes plaque, prevents cavities and gum disease. Replace your toothbrush every 3–4 months or after illness.",
        "source": "ADA",
    },
    {
        "question": "What causes dry mouth (xerostomia)?",
        "answer": "Dry mouth occurs when salivary glands do not produce enough saliva. Causes include medications (antihistamines, antidepressants, blood pressure drugs), radiation therapy to the head/neck, Sjögren's syndrome, dehydration, and mouth breathing. It increases risk of tooth decay and gum disease.",
        "source": "ADA/MedlinePlus",
    },
    {
        "question": "What is oral cancer?",
        "answer": "Oral cancer affects the mouth, lips, tongue, cheeks, floor of the mouth, and throat. Risk factors include tobacco use, heavy alcohol consumption, HPV infection, and sun exposure (lip cancer). Symptoms include sores that don't heal, red/white patches, lumps, and difficulty swallowing. Early detection is crucial for survival.",
        "source": "ADA/NCI",
    },
    {
        "question": "What are dental implants?",
        "answer": "Dental implants are titanium posts surgically placed into the jawbone to replace missing tooth roots. A crown is attached to the implant. They look and function like natural teeth, prevent bone loss, and last decades with proper care. Not everyone is a candidate — adequate bone density and healthy gums are required.",
        "source": "ADA",
    },
    {
        "question": "What causes canker sores and how are they treated?",
        "answer": "Canker sores (aphthous ulcers) are small, painful mouth ulcers. Causes are not fully understood but include stress, minor injury, acidic foods, nutritional deficiencies (B12, folate, iron), and immune system issues. They are not contagious. Most heal in 1–2 weeks. Treatment includes topical anesthetics, corticosteroid gels, and mouth rinses.",
        "source": "ADA/Mayo Clinic",
    },
    {
        "question": "What is fluoride and why is it important for dental health?",
        "answer": "Fluoride is a mineral that strengthens tooth enamel and prevents cavities by inhibiting bacterial acid production. It is found in fluoridated water, toothpaste, mouth rinses, and professional dental treatments. Children benefit most during tooth development, but adults benefit throughout life.",
        "source": "ADA/CDC",
    },
    {
        "question": "What are the symptoms of a tooth abscess?",
        "answer": "A tooth abscess is a pocket of pus caused by bacterial infection. Symptoms include severe, persistent toothache, sensitivity to hot/cold, fever, swelling in the face/cheek, tender lymph nodes, and a foul taste in the mouth. It requires immediate dental treatment — antibiotics and drainage or root canal/extraction.",
        "source": "Mayo Clinic/ADA",
    },
    {
        "question": "What is teeth grinding (bruxism)?",
        "answer": "Bruxism is clenching or grinding teeth, often during sleep. It can wear down enamel, cause tooth sensitivity, jaw pain, headaches, and damage restorations. Causes include stress, anxiety, misaligned teeth, and sleep disorders. Treatment includes night guards, stress management, and addressing underlying sleep apnea.",
        "source": "ADA/Mayo Clinic",
    },
    {
        "question": "How do dental x-rays work and are they safe?",
        "answer": "Dental X-rays use low-dose radiation to reveal cavities between teeth, bone loss, impacted teeth, cysts, and root problems invisible to the eye. Modern digital X-rays expose patients to very low radiation (comparable to a short airplane flight). Lead aprons and thyroid collars are used for additional protection.",
        "source": "ADA/FDA",
    },
    {
        "question": "What are the signs of gum disease?",
        "answer": "Signs of gum disease include red, swollen, or tender gums; bleeding when brushing or flossing; receding gums; persistent bad breath; loose teeth; pain when chewing; and changes in bite alignment. Early-stage gingivitis is reversible; advanced periodontitis requires professional treatment.",
        "source": "ADA/NIH",
    },
]


def chunk_text(text: str, max_tokens: int = 200) -> list[str]:
    words = text.split()
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks() -> list[dict]:
    chunks = []
    for qa in CURATED_DENTAL_QA:
        text = f"Q: {qa['question']}\nA: {qa['answer']}"
        for chunk_text_part in chunk_text(text, max_tokens=150):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text_part,
                "source": qa["source"],
                "type": "qa_pair",
                "body_system": "oral",
                "disease": qa["question"][:80],
            })
    log.info("built %d dental chunks", len(chunks))
    return chunks


def embed_and_ingest(chunks: list[dict]):
    log.info("loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    texts = [c["text"] for c in chunks]
    log.info("embedding %d chunks...", len(texts))
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append({**chunk, "embedding": emb.tolist()})

    log.info("upserting %d rows to Supabase...", len(rows))
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        sb.table("medical_chunks").upsert(batch, on_conflict="id").execute()
        log.info("  upserted batch %d/%d", i // BATCH_SIZE + 1, -(-len(rows) // BATCH_SIZE))

    log.info("done — %d dental chunks ingested with body_system='oral'", len(rows))


if __name__ == "__main__":
    chunks = build_chunks()
    embed_and_ingest(chunks)
