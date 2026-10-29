"""Offline routing test: DatasetRouter + merge logic, no Ollama required.
Implements the EXACT merge rules from Orchestrator.classify() so we can validate
on pure Python without Metal OOM side-effects from the embedding library."""
import sys, os
sys.path.insert(0, '.')
import numpy as np
from sentence_transformers import SentenceTransformer
from core.datasets import DatasetRouter
from core.orchestrator import (
    EmbeddingClassifier, cosine_similarity,
    KNN_TASK_OVERRIDE_THRESHOLD, KNN_TASK_OVERRIDE_WRITING_THRESHOLD,
)

emb = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
dr = DatasetRouter(emb); dr.build()
ec = EmbeddingClassifier(emb)
print("Offline router ready: %d anchors, domain_relevance_cos_gate=%.2f\n" % (
    len(dr._texts), dr.domain_relevance_cos_gate))


def classify_merge(query, image_path=None):
    qvec = emb.encode(query, normalize_embeddings=True)
    if image_path:
        return "vision", "image_detected", None

    # Step A dataset kNN
    ds_res = dr.analyse(qvec)
    if ds_res.get("ready"):
        t, tc = ds_res.get("task"), ds_res.get("task_conf", 0.0)
        cts = ds_res.get("top_task_counts")
        cos_max = ds_res.get("top_cosine_max", 0)
        parts = [f"{l}={n}" for l, n in sorted(cts.items(), key=lambda x: -x[1])]
        print(f"[dataset-knn] task vote: {t or 'ambiguous'} ({tc:.2f}) "
              f"| neighbours: {', '.join(parts)} | cos max={cos_max:.3f}"
              f"{' [OUTSIDE DOMAIN]' if t is None else ''}")

    # Step B embedding classifier (base_task = low-profile mode)
    base_task = ec.classify(qvec, allow_vision=(image_path is not None))
    base_method = "embedding"

    # Step C EXACTLY mirrors Orchestrator.classify merge
    final_task, final_method = base_task, base_method
    if (ds_res.get("ready")
            and ds_res.get("task") is not None
            and ds_res.get("task_conf", 0) >= KNN_TASK_OVERRIDE_THRESHOLD):
        ds_task = ds_res["task"]
        ds_conf = ds_res["task_conf"]
        ds_cos_max = ds_res.get("top_cosine_max", 0.0)
        safe_labels_noise = {"vision", "general", "math", "coding"}
        has_image = image_path is not None
        if ds_task == "math" and base_task in safe_labels_noise:
            final_task, final_method = "math", "dataset-knn"
        elif ds_task == "coding":
            if base_task in safe_labels_noise:
                final_task, final_method = "coding", "dataset-knn-strong"
            elif (base_task == "writing"
                  and ds_conf >= KNN_TASK_OVERRIDE_WRITING_THRESHOLD
                  and ds_cos_max >= 0.45):
                final_task, final_method = "coding", "dataset-knn-strong"
        _ = has_image
    return final_task, final_method, ds_res if ds_res.get("ready") else None


# Expected routing for each test
expectations = {
    "what is 2+3": ("math", ["easy", "medium"]),
    "can you give code for a responsive website": ("coding", ["easy", "medium", "hard"]),
    "draft an email to my team about the new responsive design guidelines": ("writing", None),
    "find the longest palindromic substring in a string": ("coding", ["medium", "hard"]),
    "write a function that reverses a list in python": ("coding", ["easy"]),
    "evaluate the integral from zero to pi of sine squared x dx": ("math", ["hard", "medium"]),
    "explain quantum entanglement in one paragraph": ("general", None),
}

all_ok = True
for q, (exp_task, exp_diffs) in expectations.items():
    print("=" * 72)
    print("QUERY:", q)
    task, method, ds = classify_merge(q)
    diff = ds.get("diff") if ds else None
    ok = (task == exp_task) and (exp_diffs is None or diff in exp_diffs)
    if not ok:
        all_ok = False
    print(f"  FINAL: [{method}] → {task} | difficulty={diff} | EXPECTED={exp_task} diff_in={exp_diffs} {'✅' if ok else '❌ WRONG'}")
    print()

print("\n" + ("ALL EXPECTATIONS MET ✅" if all_ok else "SOME ROUTING FAILURES DETECTED ❌"))
