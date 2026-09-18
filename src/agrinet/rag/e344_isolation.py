"""Resolve evaluation lineage and audit pHash <=6 across train/evaluation."""
from collections import defaultdict
from pathlib import Path
from agrinet.rag.e344_prepare import rows, digest
from agrinet.rag.e344_pipeline import write


def audit_isolation(config):
    metadata = {r['image_sha256']: r for r in rows(config['candidate_pool'])}
    evaluations = [r for path in config['evaluation_manifests'] for r in rows(path)]
    training = [r for r in metadata.values() if r.get('sft_eligible') and r.get('image_split') == 'train_candidate' and r.get('class_role') == 'known']
    unresolved, resolved, conflicts = [], [], []
    segments = [(i*256//7, (i+1)*256//7) for i in range(7)]
    buckets = defaultdict(list)
    for row in training:
        phash = row.get('phash', '')
        if len(phash) != 256 or set(phash) - {'0','1'}:
            unresolved.append({'image_sha256': row['image_sha256'], 'split': 'train', 'reason': 'invalid_phash'})
            continue
        for i,(start,end) in enumerate(segments):
            buckets[i,phash[start:end]].append(row['image_sha256'])
    source_to_sha = defaultdict(list)
    for row in training:
        if row.get('source_path'):
            source_to_sha[row['source_path']].append(row['image_sha256'])
    for public in evaluations:
        sha = public['image_sha256']
        row = metadata.get(sha)
        if not row or not row.get('source_path') or len(row.get('phash','')) != 256:
            unresolved.append({'image_sha256': sha, 'split': 'evaluation', 'reason': 'unresolved_source_or_phash'})
            continue
        resolved.append({'image_sha256': sha, 'source_path': row['source_path'], 'phash': row['phash']})
        candidates = set()
        for i,(start,end) in enumerate(segments):
            candidates.update(buckets.get((i,row['phash'][start:end]), []))
        value = int(row['phash'],2)
        for candidate in candidates:
            distance = (value ^ int(metadata[candidate]['phash'],2)).bit_count()
            if distance <= 6:
                conflicts.append({'evaluation_sha256': sha, 'training_sha256': candidate, 'phash_distance': distance})
        for candidate in source_to_sha.get(row['source_path'], []):
            conflicts.append({'evaluation_sha256': sha, 'training_sha256': candidate, 'reason': 'same_source_path'})
    result = {'threshold': 6, 'method': '256-bit pHash Hamming distance, exhaustive via seven-segment index; source path equality',
              'candidate_pool_sha256': digest(config['candidate_pool']), 'evaluations': len(evaluations),
              'resolved_evaluations': len(resolved), 'unresolved': unresolved, 'conflicts': conflicts,
              'excluded_training_shas': sorted({r['training_sha256'] for r in conflicts}),
              'limitations': 'Perceptual hashes cannot prove absence of all visual duplicates; source path identity is not acquisition-session identity.'}
    write(Path(config['output_root'])/'isolation-audit.json',result)
    return result
